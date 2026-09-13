# GPU-free tests of the remote pci device classes against a python stand-in for the TinyGPU server (unix socket, fd passing)
import unittest, os, socket, struct, threading, tempfile, mmap, contextlib
from tinygrad.helpers import unwrap
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.system import RemoteCmd, REMOTE_REQ, REMOTE_RESP, APLRemotePCIDevice, RemoteMMIOInterface, PCIIfaceBase
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.device import Buffer, BufferSpec

def resp(resp0=0, resp1=0, status=0): return struct.pack(REMOTE_RESP, status, resp0, resp1)

class MockTinyGPUServer(threading.Thread):
  """server.c in python: bars are bytearrays, config space is a bytearray, dma memory is a temp file whose fd is passed to the client"""
  def __init__(self, path:str, bars:dict[int, int]|None=None, max_sysmem:int=128):
    super().__init__(daemon=True)
    self.path, self.bars = path, {i: bytearray(sz) for i, sz in (bars or {0: 0x20000, 1: 0x40000, 3: 0x10000}).items()}
    self.cfg, self.max_sysmem, self.sysmem, self.resets, self.ops = bytearray(4096), max_sysmem, [], 0, []
    self.cfg[0:4], self.cfg[4:6] = struct.pack('<HH', 0x10de, 0x2b85), struct.pack('<H', 0x0007)
    self.cfg[0x10:0x14], self.cfg[0x14:0x1c] = struct.pack('<I', 0xa0000000), struct.pack('<Q', 0x6000000004) # bar0 32 bit, bar1 64 bit
    self.cfg[0x1c:0x24] = struct.pack('<Q', 0x7000000004)
    self.iova_next = 0x8000_0000
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with contextlib.suppress(FileNotFoundError): os.unlink(path)
    self.sock.bind(path)
    self.sock.listen(1)

  def run(self):
    while True:
      try: conn, _ = self.sock.accept()
      except OSError: return
      with conn:
        try: self.serve(conn)
        except (ConnectionError, OSError): pass

  def serve(self, conn:socket.socket):
    while len(hdr:=conn.recv(struct.calcsize(REMOTE_REQ), socket.MSG_WAITALL)) == struct.calcsize(REMOTE_REQ):
      cmd, dev_id, bar, arg0, arg1, arg2 = struct.unpack(REMOTE_REQ, hdr)
      self.ops.append(cmd)
      if cmd == RemoteCmd.MAP_BAR: conn.sendall(resp(0x100000000 + bar * 0x10000000, len(self.bars[bar])) if bar in self.bars else resp(status=1))
      elif cmd == RemoteCmd.CFG_READ: conn.sendall(resp(int.from_bytes(self.cfg[arg0:arg0+arg1], 'little')))
      elif cmd == RemoteCmd.CFG_WRITE:
        self.cfg[arg0:arg0+arg1] = arg2.to_bytes(arg1, 'little')
        conn.sendall(resp())
      elif cmd == RemoteCmd.RESET:
        self.resets += 1
        conn.sendall(resp())
      elif cmd == RemoteCmd.RESIZE_BAR: conn.sendall(resp())
      elif cmd == RemoteCmd.MMIO_READ:
        if bar not in self.bars or arg0 + arg1 > len(self.bars[bar]): conn.sendall(resp(status=1)); continue
        conn.sendall(resp(arg1) + bytes(self.bars[bar][arg0:arg0+arg1]))
      elif cmd == RemoteCmd.MMIO_WRITE:
        data = conn.recv(arg1, socket.MSG_WAITALL)
        if bar in self.bars and arg0 + arg1 <= len(self.bars[bar]): self.bars[bar][arg0:arg0+arg1] = data
      elif cmd == RemoteCmd.MAP_SYSMEM_FD:
        if len(self.sysmem) >= self.max_sysmem: conn.sendall(resp(status=1)); continue
        size = max((arg0 + 0xfff) & ~0xfff, 0x4000)
        f = tempfile.TemporaryFile()
        os.ftruncate(f.fileno(), size)
        with mmap.mmap(f.fileno(), size) as m: m[:32] = struct.pack('<4Q', self.iova_next, size, 0, 0) # one segment, like a dart
        self.sysmem.append((f, size, self.iova_next))
        self.iova_next += size
        socket.send_fds(conn, [resp(size, len(self.sysmem) - 1)], [f.fileno()])
      else:
        err = f"unknown command {cmd}".encode()
        conn.sendall(resp(len(err), status=1) + err)

  def close(self):
    self.sock.close()
    with contextlib.suppress(FileNotFoundError): os.unlink(self.path)

class FakeMM: # the memory manager side of PCIIfaceBase, records mappings
  va_base, va_bits = 0x1000000000, 44
  def __init__(self): self.next_va, self.mapped, self.unmapped = self.va_base, [], []
  def alloc_vaddr(self, size, align=0x1000):
    self.next_va = (self.next_va + align - 1) // align * align
    va, self.next_va = self.next_va, self.next_va + size
    return va
  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False):
    from tinygrad.runtime.support.memory import VirtMapping
    self.mapped.append((vaddr, size, paddrs))
    return VirtMapping(vaddr, size, paddrs, aspace=aspace, uncached=uncached, snooped=snooped)
  def unmap_range(self, vaddr, size): self.unmapped.append((vaddr, size))

class TestRemotePCI(unittest.TestCase):
  @classmethod
  def setUpClass(cls): # one server and one socket path for the process: getenv is cached
    cls.server = MockTinyGPUServer(path:=os.path.join(tempfile.mkdtemp(), "tinygpu.sock"))
    cls.server.start()
    os.environ["APL_REMOTE_SOCK"] = path

  @classmethod
  def tearDownClass(cls):
    cls.server.close()
    del os.environ["APL_REMOTE_SOCK"]

  def setUp(self):
    self.server.ops.clear()
    self.server.iova_next = 0x8000_0000
    self.dev = APLRemotePCIDevice("NV", "10de:2b85")

  def tearDown(self):
    self.dev.sock.close()
    os.close(self.dev.lock_fd)

  def test_config(self):
    self.assertEqual(self.dev.read_config(0, 4), 0x2b8510de)
    self.assertEqual(self.dev.read_config(2, 2), 0x2b85)
    self.dev.write_config(4, 0x0407, 2)
    self.assertEqual(self.dev.read_config(4, 2), 0x0407)
    self.dev.write_config_flush(4, 0x0007, 2)
    self.assertEqual(self.server.cfg[4:6], b"\x07\x00")

  def test_bar_info_and_cfg_base(self):
    self.assertEqual(self.dev.bar_info(1), (0x110000000, 0x40000))
    dev2 = APLRemotePCIDevice.__new__(APLRemotePCIDevice) # reuse the socket, bar_info is cached per instance
    dev2.__dict__.update(self.dev.__dict__)
    dev2.bar_from_cfg = True
    self.assertEqual(dev2.bar_info(1), (0x6000000000, 0x40000))
    self.assertEqual(dev2.bar_info(0)[0], 0xa0000000)

  def test_mmio(self):
    regs = self.dev.map_bar(0, fmt='I')
    self.assertIsInstance(regs, RemoteMMIOInterface)
    self.assertEqual((regs.addr, regs.residx, len(regs)), (0, 0, 0x20000 // 4))
    regs[0x10] = 0xdeadbeef
    self.assertEqual(regs[0x10], 0xdeadbeef)
    self.assertEqual(self.server.bars[0][0x40:0x44], b"\xef\xbe\xad\xde")
    regs[0x20:0x23] = [1, 2, 3]
    self.assertEqual(regs[0x20:0x23], [1, 2, 3])
    v = regs.view(0x80, 0x10)
    self.assertEqual((v.addr, len(v)), (0x80, 4))
    v[1] = 7
    self.assertEqual(regs[0x21], 7)
    raw = self.dev.map_bar(1, off=0x1000, size=0x100)
    self.assertEqual(raw.addr, 0x1000)
    raw[0:4] = b"abcd"
    self.assertEqual(bytes(raw[0:4]), b"abcd") # writes are posted: the read orders them before the check below
    self.assertEqual(bytes(self.server.bars[1][0x1000:0x1004]), b"abcd")
    self.assertEqual(self.dev.map_bar(0, fmt='I', off=0xbb0000, size=0x10000).addr + 0x90, 0xbb0090) # the doorbell's address

  def test_sysmem(self):
    self.dev.chunk_size, n0 = 1 << 20, len(self.server.sysmem)
    view, paddrs = self.dev.alloc_sysmem(0x3000)
    self.assertIsInstance(view, MMIOInterface)
    self.assertEqual(len(paddrs), 3)
    self.assertEqual(paddrs, [0x80000000, 0x80001000, 0x80002000])
    view[0:4] = b"tiny"
    self.assertEqual(len(self.server.sysmem), n0 + 1)
    with mmap.mmap(self.server.sysmem[n0][0].fileno(), 0x3000) as m: self.assertEqual(m[0:4], b"tiny") # shared with the server
    view2, paddrs2 = self.dev.alloc_sysmem(0x1000)
    self.assertEqual(paddrs2, [0x80003000])
    self.assertEqual(self.dev.sysmem_paddrs(view.addr + 0x1000, 0x2000), paddrs[1:])
    with self.assertRaises(RuntimeError): self.dev.sysmem_paddrs(0x1234000, 0x1000)
    self.dev.free_sysmem(view)
    view3, paddrs3 = self.dev.alloc_sysmem(0x2000)
    self.assertEqual(paddrs3, paddrs[:2]) # recycled
    big, big_paddrs = self.dev.alloc_sysmem(2 << 20) # bigger than a chunk: a chunk of its own
    self.assertEqual((len(self.server.sysmem) - n0, len(big_paddrs)), (2, 512))
    self.dev.free_sysmem(self.dev.map_bar(1)) # a bar window frees nothing
    self.assertEqual(self.server.ops.count(RemoteCmd.MAP_SYSMEM_FD), 2)

  def test_iface_alloc_map_free(self):
    iface = PCIIfaceBase.__new__(PCIIfaceBase)
    iface.pci_dev, iface.dev_impl, iface.vram_bar, iface.dev = self.dev, type("Impl", (), {"mm": FakeMM()})(), 1, None
    self.assertFalse(iface.is_local())
    self.assertFalse(iface.is_bar_small()) # the mock's bar1 is tiny
    st = iface.alloc(0x2000, host=True) # host allocations round to the host page size (16 KB on apple silicon)
    self.assertEqual(st.buf, 0x1000000000)
    self.assertEqual(st.meta.mapping.paddrs, [(0x80000000 + i * 0x1000, 0x1000) for i in range(mmap.PAGESIZE // 0x1000)])
    self.assertEqual(st.meta.hMemory, 0x80000000)
    unwrap(st.host)[0:4] = b"host"
    iface.free(st) # frees back to the chunk, no munmap of the gpu va
    st2 = iface.alloc(0x1000, host=True)
    self.assertEqual(st2.meta.hMemory, 0x80000000)
    # a cpu buffer over dma memory maps at a va of its own, any other cpu buffer does not map
    stg, _ = self.dev.alloc_sysmem(0x4000)
    b = Buffer("CPU", 0x4000, dtypes.uint8, options=BufferSpec(external_ptr=stg.addr), preallocate=True)
    m = iface.map(b)
    self.assertEqual((m.buf, m.meta[1]), (iface.dev_impl.mm.mapped[-1][0], 0x4000))
    self.assertEqual([p for p, _ in iface.dev_impl.mm.mapped[-1][2]], self.dev.sysmem_paddrs(stg.addr, 0x4000))
    iface.unmap(m)
    self.assertEqual(iface.dev_impl.mm.unmapped[-1], (m.buf, 0x4000))
    with self.assertRaises(RuntimeError): iface.map(Buffer("CPU", 0x2000, dtypes.uint8, preallocate=True))

  def test_compiled_mmio_write(self):
    # the host program of an hcq2 batch writes MMIO_WRITE packets with write(2): build one by hand and run it through link and clang
    from tinygrad.runtime.support.hcq2 import ccall, patch, lower_call, hcq_link, HCQInfo
    from tinygrad.engine.realize import lower_and_compile, run_linear
    from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, KernelInfo
    from tinygrad.runtime.autogen import libc
    from tinygrad.device import Device
    src = Buffer("CPU", 2, dtypes.uint32, initial_value=struct.pack('<II', 0x1234, 0xcafef00d)) # [dword index, value]: runtime inputs
    cpu, orig = Device["CPU"], Device["CPU"].pm_bufferize
    cpu.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag="apl_src"), lambda ctx, b=src: b)]) + orig # bound at link, like the fifo buffers
    self.addCleanup(setattr, cpu, "pm_bufferize", orig)
    src_p = UOp.placeholder((2,), dtypes.uint32, device="CPU", volatile=True, tag="apl_src")
    idx, val = (src_p.index(i).load() for i in range(2))
    hdr = struct.pack(REMOTE_REQ, RemoteCmd.MMIO_WRITE, 0, 0, 0, 4, 0)
    pkt = UOp.placeholder((len(hdr) + 4,), dtypes.uint8, device="CPU", volatile=True, tag="apl_test")
    pkt = patch(pkt, [(9, idx.cast(dtypes.uint64) * 4 + 0x1000), (len(hdr), val)], hdr)
    ret = UOp.placeholder((1,), dtypes.int64, device="CPU", tag="apl_ret")
    written = ret.index(0).store(ccall(libc.write, self.dev.sock.fileno(), pkt.index(0), UOp.const(len(hdr) + 4, dtypes.uint64)))
    call = UOp.sink(written, arg=KernelInfo("apl_test")).call(aux=HCQInfo(("CPU",)))
    linear = lower_and_compile(UOp(Ops.LINEAR, src=(unwrap(lower_call(call)),)))
    run_linear(hcq_link(linear, allow_cache=False), jit=True, update_stats=False, wait=True)
    self.assertEqual(self.dev.map_bar(0, fmt='I')[(0x1000 + 0x1234 * 4) // 4], 0xcafef00d)
    self.assertLess(self.server.ops.index(RemoteCmd.MMIO_WRITE), self.server.ops.index(RemoteCmd.MMIO_READ)) # the program wrote, then we read

if __name__ == "__main__":
  unittest.main()

# Diffusion-Benchmark

## Test condition:

```python

model.eval()

# warmup
for i in range(1000):    
    with torch.no_grad():
        out = model.forward()

# timed
for i in range(1000):
    with torch.no_grad():
        out = model.forward()
```

## Minimal Requirement

20 diffuse steps, 30 Hz

maximum

(1 / 30) / 20 * 1000 = 1.66 s / kitr

## Benchmark

| Device         | Method              | Performance (s / kitr) | Equiv. Rate (Hz) |
| -------------- | ------------------- | ----------- | ----------- |
| RTX 2080 [2]   | TensorRT            | 0.5         | 101.3       |
| i9 9900k [1]   | ONNX                | 2.1         |  23.9       |
| RTX 2080       | ONNX                | 2.2         |  23.2       |
| RTX 2080       | PyTorch - native    | 3.5         |  14.4       |
| i9 9900k       | PyTorch - native    | 4.0         |  12.5       |
| i9 9900k       | PyTorch - compiled  | 4.1         |  12.1       |


## i9 CPU with ONNX

Original Model: 2.8 s / kitr

Int 8 Model: 1.7 s / kitr


[1]

```yaml
Architecture:           x86_64
  CPU op-mode(s):       32-bit, 64-bit
  Address sizes:        39 bits physical, 48 bits virtual
  Byte Order:           Little Endian
CPU(s):                 16
  On-line CPU(s) list:  0-15
Vendor ID:              GenuineIntel
  Model name:           Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
    CPU family:         6
    Model:              158
    Thread(s) per core: 2
    Core(s) per socket: 8
    Socket(s):          1
    Stepping:           13
    CPU max MHz:        5000.0000
    CPU min MHz:        800.0000
    BogoMIPS:           7200.00
    Flags:              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_
                        tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl smx est tm2 ssse3 sdbg fma cx16 xtpr pd
                        cm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb
                         stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida ar
                        at pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities
Caches (sum of all):    
  L1d:                  256 KiB (8 instances)
  L1i:                  256 KiB (8 instances)
  L2:                   2 MiB (8 instances)
  L3:                   16 MiB (1 instance)
NUMA:                   
  NUMA node(s):         1
  NUMA node0 CPU(s):    0-15
Vulnerabilities:        
  Gather data sampling: Mitigation; Microcode
  Itlb multihit:        KVM: Mitigation: VMX unsupported
  L1tf:                 Not affected
  Mds:                  Not affected
  Meltdown:             Not affected
  Mmio stale data:      Mitigation; Clear CPU buffers; SMT vulnerable
  Retbleed:             Mitigation; Enhanced IBRS
  Spec rstack overflow: Not affected
  Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:           Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
  Srbds:                Mitigation; Microcode
  Tsx async abort:      Mitigation; TSX disabled
```

[2]

```
GPU 00000000:02:00.0
    Product Name                          : NVIDIA GeForce RTX 2080
    Product Brand                         : GeForce
    Product Architecture                  : Turing
    Display Mode                          : Disabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : N/A
        Pending                           : N/A
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : N/A
    GPU UUID                              : GPU-3965f0d5-5188-6a1a-0e3f-14666ad9ce20
    Minor Number                          : 1
    VBIOS Version                         : 90.04.0B.80.04
    MultiGPU Board                        : No
    Board ID                              : 0x200
    Board Part Number                     : N/A
    GPU Part Number                       : 1E87-400-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G001.0000.02.04
        OEM Object                        : 1.1
        ECC Object                        : N/A
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : N/A
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : N/A
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x02
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x1E8710DE
        Bus Id                            : 00000000:02:00.0
        Sub System Id                     : 0x86601043
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 1
                Device Current            : 1
                Device Max                : 3
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 8x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : 25 %
    Performance State                     : P8
    Clocks Event Reasons
        Idle                              : Not Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 8192 MiB
        Reserved                          : 217 MiB
        Used                              : 8 MiB
        Free                              : 7965 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 4 MiB
        Free                              : 252 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : N/A
        Pending                           : N/A
    ECC Errors
        Volatile
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
        Aggregate
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows                         : N/A
    Temperature
        GPU Current Temp                  : 27 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 100 C
        GPU Slowdown Temp                 : 97 C
        GPU Max Operating Temp            : 88 C
        GPU Target Temperature            : 83 C
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A
    GPU Power Readings
        Power Draw                        : 18.31 W
        Current Power Limit               : 215.00 W
        Requested Power Limit             : 215.00 W
        Default Power Limit               : 215.00 W
        Min Power Limit                   : 105.00 W
        Max Power Limit                   : 269.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 300 MHz
        SM                                : 300 MHz
        Memory                            : 405 MHz
        Video                             : 540 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 2130 MHz
        SM                                : 2130 MHz
        Memory                            : 7000 MHz
        Video                             : 1950 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : N/A
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1394
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 4 MiB
```
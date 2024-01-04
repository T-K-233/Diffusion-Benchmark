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

| Device           | Method              | Performance (s / kitr) | Equiv. Rate (Hz) |
| ---------------- | ------------------- | ----------- | ----------- |
| RTX 4090ᴮ [6]    | TensorRT            | 0.4         | 129.0       |
| RTX 4090ᴬ        | TensorRT            | 0.5         | 111.0       |
| RTX 2080 [2]     | TensorRT            | 0.5         | 101.3       |
| i9-9880H         | ONNX-int8           | 1.7         |  29.4       |
| GTX 1070 [4]     | TensorRT            | 1.9         |  26.0       |
| i9 9900k [1]     | ONNX                | 2.1         |  23.9       |
| RTX 2080         | ONNX                | 2.2         |  23.2       |
| i9-9880H         | ONNX                | 2.8         |  17.8       |
| RTX 2080         | PyTorch - native    | 3.5         |  14.4       |
| RTX 4090ᴮ        | PyTorch - native    | 3.6         |  13.8       |
| i9 9900k         | PyTorch - native    | 4.0         |  12.5       |
| i9 9900k         | PyTorch - compiled  | 4.1         |  12.1       |
| Ryzen 7 1700 [3] | ONNX                | 4.4         |  11.3       |
| GTX 1070         | ONNX                | 4.7         |  10.7       |
| GTX TITAN X [5]  | ONNX                | 4.7         |  10.7       |
| GTX 1070         | PyTorch - native    | 4.7         |  10.6       |
| GTX TITAN X      | PyTorch - native    | 4.8         |  10.4       |
| RTX 4090ᴬ        | PyTorch - native    | 5.2         |   9.6       |
| Ryzen 7 1700     | PyTorch - native    | 7.9         |   6.3       |
| Ryzen 7 1700     | PyTorch - compiled  | 8.4         |   6.0       |
| GTX TITAN X      | TensorRT            | NA (unsupported) | NA     |
| RTX 4090ᴬ        | ONNX                | NA (unsupported) | NA     |
| RTX 4090ᴮ        | ONNX                | NA (unsupported) | NA     |
| K510             | KPU                 | TBD         | TBD         |
| K230             | KPU                 | TBD         | TBD         |






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

```yaml
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



[3]
```yaml
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:                      43 bits physical, 48 bits virtual
CPU(s):                             16
On-line CPU(s) list:                0-15
Thread(s) per core:                 2
Core(s) per socket:                 8
Socket(s):                          1
NUMA node(s):                       1
Vendor ID:                          AuthenticAMD
CPU family:                         23
Model:                              1
Model name:                         AMD Ryzen 7 1700 Eight-Core Processor
Stepping:                           1
Frequency boost:                    enabled
CPU MHz:                            1316.086
CPU max MHz:                        3000.0000
CPU min MHz:                        1550.0000
BogoMIPS:                           5988.46
Virtualization:                     AMD-V
L1d cache:                          256 KiB
L1i cache:                          512 KiB
L2 cache:                           4 MiB
L3 cache:                           16 MiB
NUMA node0 CPU(s):                  0-15
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Mitigation; untrained return thunk; SMT vulnerable
Vulnerability Spec rstack overflow: Mitigation; safe RET, no microcode
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Retpolines, STIBP disabled, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc re
                                    p_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_l
                                    egacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb hw_pstate ssbd 
                                    vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 clzero irperf xsaveerptr arat npt lbrv svm_lock nrip_save tsc_scal
                                    e vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif overflow_recov succor smca sme sev
```

[4]

```yaml
Attached GPUs                             : 1
GPU 00000000:27:00.0
    Product Name                          : NVIDIA GeForce GTX 1070
    Product Brand                         : GeForce
    Product Architecture                  : Pascal
    Display Mode                          : Enabled
    Display Active                        : Enabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : N/A
    MIG Mode
        Current                           : N/A
        Pending                           : N/A
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : N/A
    GPU UUID                              : GPU-e0e86720-e330-c171-a8ac-b6d0d0a1021f
    Minor Number                          : 0
    VBIOS Version                         : 86.04.26.00.36
    MultiGPU Board                        : No
    Board ID                              : 0x2700
    Board Part Number                     : N/A
    GPU Part Number                       : 1B81-200-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G001.0000.01.03
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
        Bus                               : 0x27
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x1B8110DE
        Bus Id                            : 00000000:27:00.0
        Sub System Id                     : 0x33011462
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 1
                Device Current            : 1
                Device Max                : 3
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 21000 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : 33 %
    Performance State                     : P8
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 8192 MiB
        Reserved                          : 81 MiB
        Used                              : 289 MiB
        Free                              : 7821 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 5 MiB
        Free                              : 251 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 2 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : N/A
        OFA                               : N/A
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
            Single Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
            Double Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
        Aggregate
            Single Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
            Double Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows                         : N/A
    Temperature
        GPU Current Temp                  : 34 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 99 C
        GPU Slowdown Temp                 : 96 C
        GPU Max Operating Temp            : N/A
        GPU Target Temperature            : 83 C
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A
    GPU Power Readings
        Power Draw                        : 11.45 W
        Current Power Limit               : 190.00 W
        Requested Power Limit             : 190.00 W
        Default Power Limit               : 190.00 W
        Min Power Limit                   : 95.00 W
        Max Power Limit                   : 200.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 139 MHz
        SM                                : 139 MHz
        Memory                            : 405 MHz
        Video                             : 544 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1911 MHz
        SM                                : 1911 MHz
        Memory                            : 4004 MHz
        Video                             : 1708 MHz
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
        Process ID                        : 1028
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 84 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1358
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 107 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 2562
            Type                          : G
            Name                          : /snap/code/148/usr/share/code/code --type=gpu-process --disable-gpu-sandbox --no-sandbox --crashpad-handler-pid=2550 --enable-crash-reporter=f2f04b00-41d9-4d11-b4e7-219cae950694,no_channel --user-data-dir=/home/tk/.config/Code --gpu-preferences=WAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAAAAAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAGAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA== --shared-files --field-trial-handle=0,i,10887407592331812045,2785816324010611180,262144 --disable-features=CalculateNativeWinOcclusion,SpareRendererForSitePerProcess
            Used GPU Memory               : 94 MiB
```

[5]

```yaml
GPU 00000000:27:00.0
    Product Name                          : NVIDIA GeForce GTX TITAN X
    Product Brand                         : GeForce
    Product Architecture                  : Maxwell
    Display Mode                          : Enabled
    Display Active                        : Enabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : N/A
    MIG Mode
        Current                           : N/A
        Pending                           : N/A
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 0420115016841
    GPU UUID                              : GPU-cd5681f0-ecc3-b4ac-7734-0a117ab83bff
    Minor Number                          : 0
    VBIOS Version                         : 84.00.1F.00.01
    MultiGPU Board                        : No
    Board ID                              : 0x2700
    Board Part Number                     : N/A
    GPU Part Number                       : 17C2-400-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G001.0000.01.03
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
        Reset Required                    : N/A
        Drain and Reset Recommended       : N/A
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x27
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x17C210DE
        Bus Id                            : 00000000:27:00.0
        Sub System Id                     : 0x113210DE
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 1
                Device Current            : 1
                Device Max                : 3
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 1000 KB/s
        Rx Throughput                     : 5000 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : 22 %
    Performance State                     : P5
    Clocks Event Reasons
        Idle                              : Not Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : N/A
            HW Power Brake Slowdown       : N/A
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 12288 MiB
        Reserved                          : 83 MiB
        Used                              : 269 MiB
        Free                              : 11934 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 5 MiB
        Free                              : 251 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 1 %
        Memory                            : 1 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : N/A
        OFA                               : N/A
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
            Single Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
            Double Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
        Aggregate
            Single Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
            Double Bit            
                Device Memory             : N/A
                Register File             : N/A
                L1 Cache                  : N/A
                L2 Cache                  : N/A
                Texture Memory            : N/A
                Texture Shared            : N/A
                CBU                       : N/A
                Total                     : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows                         : N/A
    Temperature
        GPU Current Temp                  : 36 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 97 C
        GPU Slowdown Temp                 : 92 C
        GPU Max Operating Temp            : N/A
        GPU Target Temperature            : 83 C
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A
    GPU Power Readings
        Power Draw                        : 28.76 W
        Current Power Limit               : 250.00 W
        Requested Power Limit             : 250.00 W
        Default Power Limit               : 250.00 W
        Min Power Limit                   : 150.00 W
        Max Power Limit                   : 275.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 670 MHz
        SM                                : 670 MHz
        Memory                            : 810 MHz
        Video                             : 617 MHz
    Applications Clocks
        Graphics                          : 1000 MHz
        Memory                            : 3505 MHz
    Default Applications Clocks
        Graphics                          : 1000 MHz
        Memory                            : 3505 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1392 MHz
        SM                                : 1392 MHz
        Memory                            : 3505 MHz
        Video                             : 1281 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    Clock Policy
        Auto Boost                        : On
        Auto Boost Default                : On
    Voltage
        Graphics                          : N/A
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 967
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 72 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1289
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 128 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 2464
            Type                          : G
            Name                          : /snap/code/148/usr/share/code/code --type=gpu-process --disable-gpu-sandbox --no-sandbox --crashpad-handler-pid=2451 --enable-crash-reporter=f2f04b00-41d9-4d11-b4e7-219cae950694,no_channel --user-data-dir=/home/tk/.config/Code --gpu-preferences=WAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAAAAAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAGAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA== --shared-files --field-trial-handle=0,i,17515712563654493738,8408334331512379029,262144 --disable-features=CalculateNativeWinOcclusion,SpareRendererForSitePerProcess
            Used GPU Memory               : 62 MiB

```

[6]

```yaml
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GeForce RTX 4090
    Product Brand                         : GeForce
    Product Architecture                  : Ada Lovelace
    Display Mode                          : Enabled
    Display Active                        : Enabled
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
    Serial Number                         : 1322823110528
    GPU UUID                              : GPU-b9029ae5-75ac-6ded-3592-fc695a3f8c1a
    Minor Number                          : 0
    VBIOS Version                         : 95.02.47.00.01
    MultiGPU Board                        : No
    Board ID                              : 0x100
    Board Part Number                     : 900-1G136-2530-000
    GPU Part Number                       : 2684-301-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G002.0000.00.03
        OEM Object                        : 2.0
        ECC Object                        : 6.16
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
        Bus                               : 0x01
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x268410DE
        Bus Id                            : 00000000:01:00.0
        Sub System Id                     : 0x16F410DE
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 1
                Device Current            : 1
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 1000 KB/s
        Rx Throughput                     : 2000 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : 0 %
    Performance State                     : P8
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 24564 MiB
        Reserved                          : 354 MiB
        Used                              : 609 MiB
        Free                              : 23599 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 18 MiB
        Free                              : 238 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 34 %
        Memory                            : 5 %
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
        Current                           : Disabled
        Pending                           : Disabled
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
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 192 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 40 C
        GPU T.Limit Temp                  : 42 C
        GPU Shutdown T.Limit Temp         : -7 C
        GPU Slowdown T.Limit Temp         : -2 C
        GPU Max Operating T.Limit Temp    : 0 C
        GPU Target Temperature            : 83 C
        Memory Current Temp               : N/A
        Memory Max Operating T.Limit Temp : N/A
    GPU Power Readings
        Power Draw                        : 27.98 W
        Current Power Limit               : 450.00 W
        Requested Power Limit             : 450.00 W
        Default Power Limit               : 450.00 W
        Min Power Limit                   : 150.00 W
        Max Power Limit                   : 600.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 405 MHz
        Video                             : 1185 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 3105 MHz
        SM                                : 3105 MHz
        Memory                            : 10501 MHz
        Video                             : 2415 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 880.000 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1180
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 290 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1424
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 110 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3132
            Type                          : G
            Name                          : /snap/code/148/usr/share/code/code --type=gpu-process --disable-gpu-sandbox --no-sandbox --crashpad-handler-pid=3120 --enable-crash-reporter=190d6433-28c7-4cb5-a6a1-734946c14792,no_channel --user-data-dir=/home/tk/.config/Code --gpu-preferences=WAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAAAAAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAGAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA== --shared-files --field-trial-handle=0,i,7423938499341143706,8433235596509887907,262144 --disable-features=CalculateNativeWinOcclusion,SpareRendererForSitePerProcess
            Used GPU Memory               : 64 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 12636
            Type                          : C+G
            Name                          : /opt/google/chrome/chrome --type=gpu-process --crashpad-handler-pid=12599 --enable-crash-reporter=a42798d1-0130-45ce-ad3e-7ca1280ba517, --change-stack-guard-on-fork=enable --gpu-preferences=WAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAEAAAA4AAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAGAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA== --shared-files --field-trial-handle=0,i,8737357031540353906,16675022826240328171,262144 --variations-seed-version=20231218-080113.411000
            Used GPU Memory               : 60 MiB

```


ᴬ: on Ryzen 1700 X
ᴮ: on Intel i9 9900K

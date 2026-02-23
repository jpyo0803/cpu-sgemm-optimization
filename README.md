## CPU SGEMM 최적화 프로젝트

## CPU Info
model name      : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
cpu cores       : 4 (physical), 8 (logical)
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb pti ssbd ibrs ibpb stibp tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust sgx bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi md_clear flush_l1d arch_capabilities
l1d_cache       : 128 KiB
cache_alignment : 64


## Performance Comparison
| Optimization Technique | Latency (s) | Relative Speed |
|------------------------|-------------|---------|
| Baseline (BLAS)        | 0.008       | 1.0     |
| Naive (No Optimzation) | 5.2         | 0.0015  |
| Baseline + Compiler Flags (-O3, -march=native, -ffast-math) | 1.8         | 0.0044  |

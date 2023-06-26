nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true --osrt-threshold=10000 -x true torchrun --nproc_per_node 8 bench_slapo_inference.py --name llama
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report_ds -f true --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true --osrt-threshold=10000 -x true deepspeed --num_gpus 8 bench_ds_inference.py --name llama
# sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
export LD_LIBRARY_PATH=/data/local/tmp/eml_17/qnnbm.repo/artifacts/aarch64-android/lib:/data/local/tmp/eml_17/qnnbm.repo/resnet18_fp32:/vendor/dsp/cdsp:/usr/lib:/mnt/lib64:/vendor/lib64/:$LD_LIBRARY_PATH;export ADSP_LIBRARY_PATH="/data/local/tmp/eml_17/qnnbm.repo/artifacts/aarch64-android/lib/../../dsp/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/vendor/lib/rfsa/adsp;/dsp;/etc/images/dsp;";cd /data/local/tmp/eml_17/qnnbm.repo/resnet18_fp32;rm -rf output;chmod +x /data/local/tmp/eml_17/qnnbm.repo/artifacts/aarch64-android/bin/*;/data/local/tmp/eml_17/qnnbm.repo/artifacts/aarch64-android/bin/qnn-net-run --model libresnet18_fp32.so --input_list target_raw_list.txt --output_dir output --backend libQnnGpu.so --log_level error  --profiling_level basic --perf_profile high_performance;
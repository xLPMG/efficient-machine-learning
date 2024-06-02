# set serial of android device
export ANDROID_SERIAL=d4334008
# set user directory on the device
export DEVICE_USER_DIR=/data/local/tmp/eml_17

# create directory to host the model and data on device
adb shell "mkdir -p ${DEVICE_USER_DIR}/resnet18_gpu_int8"

# copy the runner and compiled model to the device
adb push ${QNN_SDK_ROOT}/bin/aarch64-android/qnn-net-run ${DEVICE_USER_DIR}/resnet18_gpu_int8/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnGpu.so ${DEVICE_USER_DIR}/resnet18_gpu_int8/
adb push model_libs/aarch64-android/libresnet18_int8.so ${DEVICE_USER_DIR}/resnet18_gpu_int8/

# set up target list on device
adb shell "touch ${DEVICE_USER_DIR}/resnet18_gpu_int8/target_raw_list.txt"
for batch in $(seq 0 9); do \
  adb shell "echo ${DEVICE_USER_DIR}/data/imagenet/raw_test/batch_size_32/inputs_${batch}.raw >> ${DEVICE_USER_DIR}/resnet18_gpu_int8/target_raw_list.txt"
done

# execute the model on the device GPU
adb shell "cd ${DEVICE_USER_DIR}/resnet18_gpu_int8; LD_LIBRARY_PATH=. ./qnn-net-run --backend libQnnGpu.so --model libresnet18_int8.so --input_list target_raw_list.txt"

# copy results from device to host
adb pull ${DEVICE_USER_DIR}/resnet18_gpu_int8/output output/gpu_int8
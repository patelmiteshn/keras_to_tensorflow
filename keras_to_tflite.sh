usage() {
    echo "$0 [options]"
    echo ""
    echo " -k/--keras_model            - Keras .h5 weights model"
    echo " -j/--keras_model_json       - Keras .json model"
    echo " -it/--input_tensor_names    - Input tensor names"
    echo " -is/--input_tensor_shapes   - Input tensor shapes"
    echo " -ot/--input_tensor_names    - Ouput tensor names"
    echo ""
}

check_vars() {
    if [ "${KERAS_MODEL}" = "" ] || \
        [ "${KERAS_MODEL}" = "" ] || \
        [ "${KERAS_MODEL_JSON}" = "" ] || \
        [ "${OUTPUT}" = "" ] || \
        [ "${OUTPUT_PB}" = "" ] || \
        [ "${INPUT_TENSOR_NAMES}" = "" ] || \
        [ "${INPUT_TENSOR_SHAPES}" = "" ] || \
        [ "${OUTPUT_TENSOR_NAMES}" = "" ]; then
        usage
        echo KERAS_MODEL = ${KERAS_MODEL}
        echo KERAS_MODEL_JSON = ${KERAS_MODEL_JSON}
        echo OUTPUT = ${OUTPUT}
        echo INPUT_TENSOR_NAMES = ${INPUT_TENSOR_NAMES}
        echo INPUT_TENSOR_SHAPES = ${INPUT_TENSOR_SHAPES}
        echo OUTPUT_TENSOR_NAMES = ${OUTPUT_TENSOR_NAMES}
        echo ""
        exit
    fi
}

for i in "$@"
do
case $i in
    -k=*|--keras_model=*)
        KERAS_MODEL="${i#*=}"
    ;;
    
    -j=*|--keras_model_json=*)
        KERAS_MODEL_JSON="${i#*=}"
    ;;
    
    -o=*|--output=*)
        OUTPUT="${i#*=}"
    ;;

    -it=*|--input_tensor_names=*)
        INPUT_TENSOR_NAMES="${i#*=}"
    ;;

    -is=*|--input_tensor_shapes=*)
        INPUT_TENSOR_SHAPES="${i#*=}"
    ;;
   

    -ot=*|--output_tensor_names=*)
        OUTPUT_TENSOR_NAMES="${i#*=}"
    ;;
    
    -h*|--help*)
            usage
            exit
    ;;

    *)
            usage
            exit
    ;;
esac
done

OUTPUT_PB=${OUTPUT%.*}.pb

check_vars

echo "*** Converting Keras -> Tensorflow ***"
python keras_to_tensorflow.py \
    --input_model ${KERAS_MODEL} \
    --input_model_json ${KERAS_MODEL_JSON} \
    --output_model ${OUTPUT_PB}

echo "*** Converting Tensorflow -> TFLite ***"
toco --graph_def_file=${OUTPUT_PB} \
  --output_format=TFLITE \
  --output_file=${OUTPUT} \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=${INPUT_TENSOR_NAMES} \
  --output_arrays=${OUTPUT_TENSOR_NAMES} \
  --input_shapes=${INPUT_TENSOR_SHAPES}
  
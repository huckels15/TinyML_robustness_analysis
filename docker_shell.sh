export IMAGE_NAME="tinyml_robustness_analysis"
export BASE_DIR=$(pwd)


echo "Building image..."
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
--mount type=bind,source="$BASE_DIR",target=/package $IMAGE_NAME
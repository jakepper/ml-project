using Images, CSV, DataFrames, DecisionTree

function read_data(typeData, labelsInfo, imageSize, path)
    # Intialize x matrix
    x = zeros(size(labelsInfo, 1), imageSize)

    for (index, idImage) in enumerate(labelsInfo[:,"ID"]) 
        # Read image file 
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
        img = load(nameFile)

        # Convert img to float values 
        temp = float32.(img)

        # Convert color images to gray images by taking the average of the color scales.
        gray = Gray.(temp)
            
        # Transform image matrix to a vector and store it in data matrix 
        x[index, :] = reshape(gray, (1, imageSize))
    end

    return x
end

# 20 x 20 pixel
imageSize = 400

# Set location of data files, folders
path = "."

# Read information about training data , IDs.
labelsInfoTrain = DataFrame(CSV.File("$(path)/trainLabels.csv"))

# Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

# Read information about test data ( IDs ).
labelsInfoTest = DataFrame(CSV.File("$(path)/sampleSubmission.csv"))

# Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

# Get only first character of string (convert from string to character).
# Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[:,"Class"])

# Convert from character to integer
yTrain = Int.(yTrain)

# Train random forest with
# 20 for number of features chosen at each random split,
# 50 for number of trees,
# and 1.0 for ratio of subsampling.
model = build_forest(yTrain, xTrain, 20, 50, 1.0)

# Get predictions for test data
predTest = apply_forest(model, xTest)

# Convert integer predictions to character
labelsInfoTest[:,"Class"] = string.(Char.(predTest))

# Save predictions
CSV.write("$(path)/juliaSubmission.csv", labelsInfoTest)

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 4, 1.0, verbose=false);
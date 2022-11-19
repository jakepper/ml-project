using Images, CSV, DataFrames, DecisionTree, Statistics

# Read training matrix
xTrain = CSV.File("./xTrain.csv") |> Tables.matrix

# Read test matrix
xTest = CSV.File("./xTest.csv") |> Tables.matrix
println(axes(xTest))

# Read yTrain
yTrain = CSV.File("./yTrain.csv") |> Tables.matrix
yTrain = vec(yTrain)

# Read information about test data ( IDs ).
labelsInfoTest = DataFrame(CSV.File("./sampleSubmission.csv"))

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
CSV.write("./juliaSubmission.csv", labelsInfoTest)

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 4, 1.0, verbose=false)
println("4 fold accuracy: $(mean(accuracy))")
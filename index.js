require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

// k nearest neighbor algorithm - https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
// The features are standarized to give all features equal weight.
// the distance of the features in the training set are calculated against all the feautres in the prediction point
// the distance of every feature is than summed together to create a total
// the total is than sorted from least to greatest. The closer a value is to 0... the more it pertains to the prediction point
// the top k values are averaged and this is our resulting prediction
function knn(features, labels, predictionPoint, k) {
    // pass the data set to tensorflow moments to calculate the mean and variance
    // of the entire dataset
    const { mean, variance } = tf.moments(features, 0);

    // calculate the standard deviation for the predictionPoint
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

    // standardize the features of the trainingset
    const standardizedFeatures = features
    .sub(mean)
    .div(variance.pow(0.5));
      
    // calculate distance between feature and scaled prefictionPointFeature
    // d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    const distance = standardizedFeatures
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(.5);
      
    // sort from least to greatest distance
    const sorted = distance.expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1);
    
    // average top values
    const average = sorted.slice(0, k)
    .reduce((acc, t) => acc + t.get(1), 0) / k;

    return average;
}

// returns the error percentage of the predicted value
// based off of the expected value
function calculateError(expected, predicted) {
    return ((expected - predicted) / expected) * 100;
}

// parse CSV data into arrays
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10, // Testing dataset - # of records
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
});

// convert arrays into tensor objects
features        = tf.tensor(features);
labels          = tf.tensor(labels);

const k = 10;

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), k);
    console.log("KNN result:", result);
    console.log("Actual answer:", testLabels[i][0]);
    const err = calculateError(testLabels[i][0], result);
    console.log("Error percentage:", err);
});
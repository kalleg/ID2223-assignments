package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.sql.{ SQLContext, DataFrame }
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.Pipeline
import se.kth.spark.lab1.DoubleUDF
import se.kth.spark.lab1.Array2Vector
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.regression.LinearRegression
import se.kth.spark.lab1.Vector2DoubleUDF
import org.apache.spark.ml.feature.PolynomialExpansion

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    // Get traning and test sets
    val obsDF: DataFrame = sqlContext.read.text(filePath)
    val Array(training, test) = obsDF.randomSplit(Array[Double](0.7, 0.3))
    
    // Transformers
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("row")
      .setPattern(",")
      
    val tokenized = regexTokenizer.transform(training)

    val arr2Vect = new Array2Vector()
      .setInputCol("row")
      .setOutputCol("vector")
    val vectorized = arr2Vect.transform(tokenized)

    val lSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("year")
    val sliced = lSlicer.setIndices(Array(0)).transform(vectorized)

    val v2d = new Vector2DoubleUDF(a => a(0))
      .setInputCol("year")
      .setOutputCol("label")
    val label = v2d.transform(sliced)
 
    label.createOrReplaceTempView("songs")
    val min = label.select("label").map(song => song.getDouble(0)).reduce((a, b) => if (a < b) a else b)
    val lShifter = new DoubleUDF(a => a - min)
      .setInputCol("label")
      .setOutputCol("labelShift")

    val fSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("features")
    val sliced2 = fSlicer.setIndices(Array(1, 2, 3))

    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("featuresExpand")
      .setDegree(2)
    
    val myLR = new LinearRegression()
      .setLabelCol("labelShift")
      .setFeaturesCol("featuresExpand")
      .setElasticNetParam(0.1)
      
    val lrStage = 7
    
    // Pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, polynomialExpansion, myLR))
 
    //build the parameter grid by setting the values for maxIter and regParam
    val paramGrid = new ParamGridBuilder()
        .addGrid(myLR.maxIter, Array(5, 10, 30, 50, 75, 100))
        .addGrid(myLR.regParam, Array(0.05, 0.1, 0.3, 0.6, 0.9, 0.95))
        .build()
    
    //create the cross validator and set estimator, evaluator, paramGrid
    val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(new RegressionEvaluator())
        .setEstimatorParamMaps(paramGrid)
    val cvModel: CrossValidatorModel = cv.fit(training)
      
    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction - print first k
    val pipelineTest = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, polynomialExpansion))
    val testSet = pipelineTest.fit(test).transform(test).select("featuresExpand", "labelShift")
    
    val predictions = lrModel.transform(testSet)
    
    predictions.map(row => row.getDouble(1)-row.getDouble(2)).show(5)
    
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
  }
}
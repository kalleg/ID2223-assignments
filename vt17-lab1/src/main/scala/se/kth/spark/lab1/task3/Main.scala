package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import se.kth.spark.lab1.DoubleUDF
import se.kth.spark.lab1.Array2Vector
import org.apache.spark.ml.feature.RegexTokenizer
import se.kth.spark.lab1.Vector2DoubleUDF
import org.apache.spark.ml.regression.LinearRegression

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

    val myLR = new LinearRegression()
      .setLabelCol("labelShift")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
      
    val lrStage = 6
    
    // Pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    val pipelineModel = pipeline.fit(training)
    
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction - print first k
    val pipelineTest = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))
    val testSet = pipelineTest.fit(test).transform(test).select("features", "labelShift")
    
    val predictions = lrModel.transform(testSet)
    
    predictions.map(row => row.getDouble(1)-row.getDouble(2)).show(5)
    
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
  }
}
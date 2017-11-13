package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorSlicer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)
    rawDF.show(5)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("row")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val tokenized = regexTokenizer.transform(rawDF)
    tokenized.show(5)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("row")
      .setOutputCol("vector")
    val vectorized = arr2Vect.transform(tokenized)
    vectorized.show(5)
    
    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("year")
    val sliced = lSlicer.setIndices(Array(0)).transform(vectorized)
    sliced.show(5)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(a => a(0))
      .setInputCol("year")
      .setOutputCol("label")
    val label = v2d.transform(sliced)
    label.show(5)
    
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    label.createOrReplaceTempView("songs")
    val min = label.select("label").map(song => song.getDouble(0)).reduce((a, b) => if (a < b) a else b)
    val lShifter = new DoubleUDF(a => a - min)
      .setInputCol("label")
      .setOutputCol("labelShift")
    val labelShift = lShifter.transform(label)
    labelShift.show(5)
    
    //Step7: extract just the 3 first features in a new vector column
    //val fSlicer = ???

    //Step8: put everything together in a pipeline
    //val pipeline = new Pipeline().setStages(???)

    //Step9: generate model by fitting the rawDf into the pipeline
    //val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    //???

    //Step11: drop all columns from the dataframe other than label and features
    //???
  }
}
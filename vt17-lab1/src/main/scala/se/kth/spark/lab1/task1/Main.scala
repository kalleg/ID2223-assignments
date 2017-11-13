package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.Row
import org.apache.spark.sql.RowFactory

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/millionsong.txt"

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    //rdd.take(5).foreach(println)
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(line => line.split(","))
    //recordsRdd.take(5).foreach(line => {line.foreach(println)})
    
    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(row => (row(0).toDouble, row(1).toDouble, row(2).toDouble, row(3).toDouble))
    //songsRdd.take(5).foreach(println)

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF("year", "feature1", "feature2", "feature3")
    //songsDf.printSchema()

    // Task 1.1
    val count = songsDf.map(song => 1).reduce((a, b) => a+b)
    println(count)
   
    songsDf.createOrReplaceTempView("songs")
    sqlContext.sql("SELECT COUNT(year) FROM songs").show()
    
    // Task 1.2
    val numOfReleases = songsDf.filter($"year" >= 1998 && $"year" <= 2000).map(song => 1).reduce((a, b) => a+b)
    println(numOfReleases)
    
    sqlContext.sql("SELECT COUNT(year) FROM songs WHERE year >= 1998 AND year <= 2000").show()
    
    // Task 1.3
    val min = songsDf.map(song => song.getDouble(0)).reduce((a, b) => if (a < b) a else b)
    println(min)
    val max = songsDf.map(song => song.getDouble(0)).reduce((a, b) => if (a > b) a else b)
    println(max)
    val sum = songsDf.map(song => song.getDouble(0)).reduce((a, b) => a+b)
    println(sum/count)
    
    sqlContext.sql("SELECT MIN(year), AVG(year), MAX(year) FROM songs").show()
    
    // Task 1.4
    songsDf.filter($"year" >= 2000 && $"year" <= 2010).groupBy("year").count().show()
    sqlContext.sql("SELECT year, COUNT(year) FROM songs WHERE year >= 2000 AND year <= 2010 GROUP BY year").show()
    
  }
}
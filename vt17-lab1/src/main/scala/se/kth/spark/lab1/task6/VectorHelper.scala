package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    var product = 0.0
    var i = 0
    for (i <- 0 until v1.size) {
      product += v1(i) * v2(i)
    }
    return product
  }
  
  def dot(v: Vector, s: Double): Vector = {
    return Vectors.dense(v.toArray.map(v => v * s))
  }
  
  def sum(v1: Vector, v2: Vector): Vector = {
    val result = new Array[Double](v1.size)
    var i = 0
    for (i <- 0 until v1.size) {
      result(i) = v1(i) + v2(i)
    }
    return Vectors.dense(result)
  }
  
  def fill(size: Int, fillVal: Double): Vector = {
    return Vectors.dense(new Array[Double](size).map(v => fillVal))
  }
}
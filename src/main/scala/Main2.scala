/*

A tensor is a array of data [512, Values], where Values can be a Vector space, a 2D-Grid, 3D-Grid, etc.
Operations are defined on Value types. E.g. A Vector space transformation, a 2D-Kernel.

*/

package main2

import util.*

import scala.util.Random
import scala.quoted.*


trait ValueType
trait Vector[Dim <: Int]() extends ValueType
trait NamedVector[ColNames <: Tuple] extends Vector[Length[ColNames]]
trait Grid2D[XDim <: Int, YDim <: Int] extends ValueType
type Scalar = Vector[1]

type NumValues <: Int


case class ValueList[NumValues, NamedVector]()

trait Relation[From <: ValueType, To <: ValueType]
case class VectorSpaceRelation[D1 <: Int, D2 <: Int]() extends Relation[Vector[D1], Vector[D2]]

object VectorSpaceOps:

  extension [Size <: Int, From <: Int, FromV <: Vector[From]](tensor: ValueList[Size, FromV])
    def dot[To <: Int](
      rel: VectorSpaceRelation[From, To]
    ): ValueList[Size, Vector[To]] = 
      new ValueList[Size, Vector[To]]()


@main def hello2(): Unit =

  import VectorSpaceOps.*
  {
    println("** Linear Model EXAMPLE **")
    val t1 = ValueList[512, NamedVector[("width", "length", "height")]]()
    // val t1 = ValueList[256, Vector[3]]()
    val weights = VectorSpaceRelation[3, 10]()
    val t2 = t1.dot(weights)
    println(t2)
  }
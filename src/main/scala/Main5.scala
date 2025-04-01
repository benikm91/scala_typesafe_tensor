package main5

import scala.compiletime.ops.int._
import java.util.jar.Attributes.Name
import scala.compiletime.{erasedValue, constValue, constValueOpt}
import scala.util.Random

trait Structure[A]
case class Branch[A](shapes: List[Structure[A]]) extends Structure[A]:
  override def toString(): String = shapes.mkString("(", ", ", ")")
case class Leaf[A](shape: A) extends Structure[A]:
  override def toString(): String = shape.toString()
object EmptyStructure extends Structure:
  override def toString(): String = "()"

case class Name(name: String)

type Shape = Structure[Int]
type NameStructure = Structure[Name]

trait Index[A]

trait NameIndex[A] extends Index[A]:
  def names: NameStructure

object NameIndex:
  
  given stringIsNameStructure[T <: String:  ValueOf]: NameIndex[T] with
    inline def names: NameStructure = Leaf(Name(summon[ValueOf[T]].value))

  given nameIsNameStructure[T <: Name:  ValueOf]: NameIndex[T] with
    inline def names: NameStructure = Leaf(summon[ValueOf[T]].value)

  given [T <: Name : NameIndex]: NameIndex[Tuple1[T]] with
    inline def names: NameStructure =  Branch(List(summon[NameIndex[T]].names))

  given [T1: NameIndex, T2: NameIndex]: NameIndex[Tuple2[T1, T2]] with
    inline def names: NameStructure = Branch(List(summon[NameIndex[T1]].names, summon[NameIndex[T2]].names))

  given [T1: NameIndex, T2: NameIndex, T3: NameIndex]: NameIndex[Tuple3[T1, T2, T3]] with
    inline def names: NameStructure = Branch(List(summon[NameIndex[T1]].names, summon[NameIndex[T2]].names, summon[NameIndex[T3]].names))


trait ShapeIndex[A] extends Index[A]:
  def shape: Shape

object ShapeIndex:

  given intIsIndex[T <: Int:  ValueOf]: ShapeIndex[T] with
    inline def shape: Shape = Leaf(summon[ValueOf[T]].value)

  given [T <: Int : ShapeIndex]: ShapeIndex[Tuple1[T]] with
    inline def shape: Shape =  Branch(List(summon[ShapeIndex[T]].shape))

  given [T1: ShapeIndex, T2: ShapeIndex]: ShapeIndex[Tuple2[T1, T2]] with
    inline def shape: Shape = Branch(List(summon[ShapeIndex[T1]].shape, summon[ShapeIndex[T2]].shape))

  given [T1: ShapeIndex, T2: ShapeIndex, T3: ShapeIndex]: ShapeIndex[Tuple3[T1, T2, T3]] with
    inline def shape: Shape = Branch(List(summon[ShapeIndex[T1]].shape, summon[ShapeIndex[T2]].shape, summon[ShapeIndex[T3]].shape))

case class Tensor[I : ShapeIndex]():
  def shape: Shape = summon[ShapeIndex[I]].shape
  override def toString(): String = f"Tensor[${shape}]"

@main def hello5(): Unit =
  val xx = summon[ShapeIndex[100]]
  println(xx.shape)
  val runtime = Random.between(1, 10)
  val runtimeIndex = summon[ShapeIndex[runtime.type]]
  println(runtimeIndex.shape)
  val tuple1Index = summon[ShapeIndex[Tuple1[100]]]
  println(tuple1Index.shape)
  val tuple2Index = summon[ShapeIndex[(100, 200)]]
  println(tuple2Index.shape)
  val tuple3Index = summon[ShapeIndex[(100, 200, 300)]]
  println(tuple3Index.shape)


  val tuple3Names = summon[NameIndex[("A", "B", "C")]]
  println(tuple3Names.names)

  val n1 = summon[NameIndex["extra"]]
  println(n1.names)

  val t1 = Tensor[100]()
  val t2 = Tensor[(100, 200)]()

  println((t1, t2))
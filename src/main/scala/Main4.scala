/*

Two kinds of dimensionens:
* Tensor dimension (Space): Tensor[(Points[100], Space[3])]()
* Space dimension (Axis): Space[3] ^= ("length", "height", "width")
** Interestring as each axis can have a name, a precision (dtype) and a unit of measure. For example, the unit of measure could be tracked through *, / operations and standardization.
** Probably to far...

Tensor library (numpy) should only take care of the tensor dimensions.
Library build on top (e.g. Dataframe library (pandas)) can take care of the space dimensions.
=> In tensor library all space dimensions must share the same dtype.

Order inside the dimensions?

*/

package main4

import scala.compiletime.ops.int._
import java.util.jar.Attributes.Name

trait Dimension
trait Points[Num <: Int] extends Dimension  // sample
trait Space[Dim <: Int] extends Dimension  // concepts

type LinearProjection[A <: Int, B <: Int] = (Space[A], Space[B])

type Grid2D[X <: Int, Y <: Int, C <: Int] = (Points[X], Points[Y], Space[C])
type Grid3D[X <: Int, Y <: Int, Z <: Int, C <: Int] = (Points[X], Points[Y], Points[Z], Space[C])
// type NamedGrid2D[X <: Int, Y <: Int, XName <: String, YName <: String] = (Points[X] & Named[XName], Points[Y] & Named[YName], Space[1])
// type PointCloud2D[NumPoints <: Int] = (Space[2], Points[NumPoints])

type Shape = Tuple

case class Tensor[S <: Shape]()

type X = Tensor[(100, 3)]

object VectorOps:

  extension [
    Ax <: Points[?], 
    From <: Int, 
    From1Ax <: Space[From],
    From2Ax <: Space[From],
  ](
    t1: Tensor[(Ax, From1Ax)]
  )

    def dot[To <: Int, ToAx <: Space[To]](
      t2: Tensor[(From2Ax, ToAx)]
    ): Tensor[(Ax, ToAx)] = 
      new Tensor[(Ax, ToAx)]()


object Grid2DOps:

  trait Convolution[C <: Int, I <: (Points[?], Points[?], Space[C]), K <: (Points[?], Points[?], Space[C])]:
    type Out <: (Points[?], Points[?], Space[?])

  object Convolution:
    type Aux[C <: Int, I <: (Points[?], Points[?], Space[C]), K <: (Points[?], Points[?], Space[C]), O <: (Points[?], Points[?], Space[1])] = Convolution[C, I, K] { type Out = O }

    given xxx[IX <: Int, IY <: Int, C <: Int, KX <: Int, KY <: Int]: Convolution.Aux[C, Grid2D[IX, IY, C], Grid2D[KX, KY, C], Grid2D[IX - KX + 1, IY - KY + 1, 1]] =
      new Convolution[C, Grid2D[IX, IY, C], Grid2D[KX, KY, C]]:
        type Out = Grid2D[IX - KX + 1, IY - KY + 1, 1]

  extension [
    C <: Int,
    V <: Points[?], 
    G <: (Points[?], Points[?], Space[C]),
  ](
    t1: Tensor[(V, G)]
  )
    def conv[
      K <: (Points[?], Points[?], Space[C])
    ](
      kernel: Tensor[K]
    )(using
      ot: Convolution[C, G, K],
    ): Tensor[(V, ot.Out)] = 
      new Tensor[(V, ot.Out)]()
 
import VectorOps.*

type BatchSize = 32

/** --- ML / MATH STUFF --- */
def MLStuff = {
  val X = Tensor[(Points[100], Space[3])]()   // 10_000 samples of 3 features
  val W1 = Tensor[(Space[3], Space[32])]()
  val W2 = Tensor[(Space[32], Space[1])]()    // project 3 features to 1 output
  // val W = W1.dot(W2)  // Space 3 -> Space 1
  val y_out = X.dot(W1).dot(W2)

  // val W = Tensor[(Space[3], Space[1])]()      // project 3 features to 1 output

  // Hat-Matrix
  val XT = Tensor[(Space[3], Points[10_000])]()               // tranpose X
  val XTX = Tensor[(Space[3], Space[3])]()                    // coverance matrix
  val XTXinv = Tensor[(Space[3], Space[3])]()                 // inverse
  val XTXinvXT = Tensor[(Space[3], Points[10_000])]()         // ?? 
  val XXTXinvXT = Tensor[(Points[10_000], Points[10_000])]()  // ?? => Hat-Matrix
  val y = Tensor[(Points[10_000], Space[1])]()
}
// val y_hat: Tensor[(Points[10_000], Space[1])] = hatMatrix * y 

// (xT * x)-1 xT y


// Given a matrix X = Tensor[(Points[10_000], Space[10])]
val coverianceMatrix = Tensor[(Space[10], Space[10])]()              // X.T * X - covariance between fefatures
val linearKernelMatrix = Tensor[(Points[10_000], Points[10_000])]()   // X * X.T - relationship between samples

val X = Tensor[(Points[4], Points[32], Points[32], Space[3])]()
val X2 = Tensor[(Points[4], Grid2D[32, 32, 3])]()

/** --- NLP Stuff --- */
// Some NLP stuff with Transformers, Vocab size 50_000, Latent Dim 512, Context Window 128
val contextTokens = Tensor[(Points[128], Points[50_000])]()           // context is 128 tokens of 50_000 words (vocabulary)
val encoder = Tensor[(Points[50_000], Space[512])]()                  // encoder encodes a word (vocabulary) to a vector space
val embeddings1 = Tensor[(Points[128], Space[512])]()                 // embeddings are 128 vectors in a vector space
val attentionMatrix = Tensor[(Points[128], Points[128])]()            // attention mask are 128 attention scores of 128 vectors
// with batch size
val embeddings2 = Tensor[(Points[BatchSize], Points[128], Space[512])]()
val attentionMatrix2 = Tensor[(Points[BatchSize], Points[128], Points[128])]()


@main def hello4(): Unit = 
  {
    import Grid2DOps.*
    val imgData = Tensor[(Points[4], Grid2D[32, 32, 3])]()
    val kernel = Tensor[Grid2D[3, 3, 3]]()
    val kernel2 = Tensor[(Points[3], Points[3], Space[3], Space[1])]()
    val singleFeatureMap = imgData.conv(kernel)
  }
  {
    import Grid2DOps.*

    // val X_2D_imgs = Tensor[(Points[4], (Points[28], Points[28], Space[1]))]()
    // val X_3D_img_ = Tensor[(Points[4],  Points[28], Points[28], Space[1] )]()
    val X_2D_imgs = Tensor[(Points[4], Grid2D[28, 28, 1])]()
    val X_3D_img_ = Tensor[(Grid3D[4, 28, 28, 1])]()
  }
  {
    // Order
    trait PointsOrder
    trait NoOrder extends PointsOrder
    trait UniformOrder extends PointsOrder  // conv requires uniform order
    trait AscOrder extends PointsOrder
    trait DescOrder extends PointsOrder
    trait UniformAscOrder extends UniformOrder with AscOrder
    trait UniformDescOrder extends UniformOrder with DescOrder
    type Grid2D[X <: Int, Y <: Int, C <: Int] = (Points[X] & UniformAscOrder, Points[Y] & UniformDescOrder, Space[C])
    val X_2D_imgs = Tensor[(Points[4], Grid2D[28, 28, 1])]()
  }
  {

    trait Named[Name <: String]
    object Named:
      trait Extract[A]:
        type Out <: Tuple

      object Extract {
        type Aux[A, Out0 <: Tuple] = Extract[A] { type Out = Out0 }

        import scala.compiletime.{constValue, erasedValue, summonFrom, summonInline}
        import scala.compiletime.ops.string.Matches

        inline def apply[A](using e: Extract[A]): Extract[A] = e

        inline given tupleExtract[A <: Tuple]: Extract.Aux[A, Tuple.Map[A, ExtractName]] =
          new Extract[A] {
            type Out = Tuple.Map[A, ExtractName]
          }

        type ExtractNames[A <: Tuple] = Tuple.Map[A, ExtractName]

        type ExtractName[A] = A match {
          case Named[n] => n
          case _ => Unit
        }
      }

    // Test Named
    val t = Tensor[(Points[32] & Named["length"], Points[32] & Named["width"])]()

    type X = (Points[32] & Named["length"], Points[32] & Named["width"], Named["height"], Points[32] & Named["asdasd"])
    type Names = Named.Extract.ExtractNames[
      X
    ]
    // val names: Names = ("length", "width", "height", ())
  }
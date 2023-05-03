package waig548.NNN.normalization

import kotlinx.serialization.Serializable
import waig548.NNN.util.math.Matrix

@Serializable
object FeatureNormalization: Normalization
{
    override fun normalize(x: Matrix<Double>, layerID: Int, training: Boolean): Matrix<Double>
    {
        TODO("Not yet implemented")
    }

    override fun update(dy: Matrix<Double>, layerID: Int): Matrix<Double>
    {
        TODO("Not yet implemented")
    }
}
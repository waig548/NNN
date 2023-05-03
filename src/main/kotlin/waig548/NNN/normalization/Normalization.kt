package waig548.NNN.normalization

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonClassDiscriminator
import waig548.NNN.util.math.Matrix

@Serializable
@JsonClassDiscriminator("type")
sealed interface Normalization
{
    fun normalize(x: Matrix<Double>, layerID: Int, training: Boolean = false): Matrix<Double>
    fun update(dy: Matrix<Double>, layerID: Int): Matrix<Double>
}
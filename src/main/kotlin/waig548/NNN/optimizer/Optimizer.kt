package waig548.NNN.optimizer

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonClassDiscriminator
import waig548.NNN.util.math.Matrix

@Serializable
@JsonClassDiscriminator("type")
sealed interface Optimizer
{
    fun update(
        iteration: Int,
        layerID: Int,
        //weights: Matrix<Double>,
        //biases: Matrix<Double>,
        dW: Matrix<Double>,
        dB: Matrix<Double>
    ): Pair<Matrix<Double>, Matrix<Double>>
}
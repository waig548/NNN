package waig548.NNN.functions

import kotlin.math.max

object ReLU : ActivationFunction
{
    override val name: String = this::class.java.simpleName
    override val derivative: (Double) -> Double = {z -> if (z>=0) 1.0 else 0.01}
    override fun invoke(z: Double): Double
    {
        return max(z, 0.01*z)
    }
}
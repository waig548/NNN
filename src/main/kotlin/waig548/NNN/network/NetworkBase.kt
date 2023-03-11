package waig548.NNN.network

import waig548.NNN.util.math.Matrix

sealed class NetworkBase
{
    abstract val size: Int

    var x: Matrix<Double> = Matrix(1) { 0.0 }
    var y: Matrix<Double> = Matrix(1) { 0.0 }
    var output: Matrix<Double> = Matrix(1) { 0.0 }

}

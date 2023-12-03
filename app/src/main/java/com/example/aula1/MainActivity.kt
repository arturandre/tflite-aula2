package com.example.aula1

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.aula1.databinding.ActivityMainBinding

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.io.FileInputStream
import java.io.IOException
import java.lang.ClassCastException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

// Initialization code

// Operations map
var opMap = mutableMapOf(
    "resize" to false,
    "crop" to false,
    "pad" to false,
    "rot" to false,
    "gray" to false,
    "norm" to false
)

val resOp = ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)
val cropOp = ResizeWithCropOrPadOp(100, 100)
val padOp = ResizeWithCropOrPadOp(500, 500)
val rotOp = Rot90Op(2) // k*90 Anti-clockwise rotation
val grayOp = TransformToGrayscaleOp()
val normOp = NormalizeOp(127f, 1f)

var opMapF = mapOf(
    "resize" to resOp,
    "crop" to cropOp,
    "pad" to padOp,
    "rot" to rotOp,
    "gray" to grayOp,
    "norm" to normOp
)

val classToName = mapOf(
    0 to "Adélie",
    1 to "Chinstrap",
    2 to "Gentoo"
)

/*var imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .build()*/




// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.


private lateinit var binding: ActivityMainBinding

fun <T : Comparable<T>> argmax(iterable: Iterable<T>): Int? {
    var iterator = iterable.iterator()

    if (!iterator.hasNext()) return null

    var maxIndex = 0
    var currentIndex = 0
    var maxValue = iterator.next()

    while (iterator.hasNext()){
        currentIndex++
        val value = iterator.next()
        if (value > maxValue){
            maxValue = value
            maxIndex = currentIndex
        }
    }

    return maxIndex
}

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //Ref: https://developer.android.com/topic/libraries/view-binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        val outputs = classifySequence()
        val cls = listOf(
            argmax(outputs[0].asIterable()),
            argmax(outputs[1].asIterable()),
            argmax(outputs[2].asIterable()),
            )
        var outputStr = ("example 0: \n"
                    + outputs[0][0].toString()
                    + " | "
                    + outputs[0][1].toString()
                    + " | "
                    + outputs[0][2].toString()
                    + "\n"
                    + "Class: " + cls[0] + " : " + classToName[cls[0]]
                )


        val cls1 = argmax(outputs[1].asIterable())
        outputStr = (outputStr + "\n\n"
                    + "example 1: \n"
                    +
                    outputs[1][0].toString()
                    + " | "
                    + outputs[1][1].toString()
                    + " | "
                    + outputs[1][2].toString()
                    + "\n"
                    + "Class: " + cls[1] + " : " + classToName[cls[1]]
                    )
        val cls2 = argmax(outputs[2].asIterable())
        outputStr = (outputStr + "\n\n"
                    + "example 2: \n"
                    +
                    outputs[2][0].toString()
                    + " | "
                    + outputs[2][1].toString()
                    + " | "
                    + outputs[2][2].toString()
                    + "\n"
                    + "Class: " + cls[2] + " : " + classToName[cls[2]]
                    )

        binding.outputLog.setText(outputStr, TextView.BufferType.EDITABLE)
    }

    //Ref:
    //https://towardsdatascience.com/spam-classification-in-android-with-tensorflow-lite-cde417e81260
    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val MODEL_ASSETS_PATH = "model.tflite"
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.getFileDescriptor())
        val fileChannel = fileInputStream.getChannel()
        val startoffset = assetFileDescriptor.getStartOffset()
        val declaredLength = assetFileDescriptor.getDeclaredLength()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    }

    //fun classifySequence ( inputs : Array<FloatArray> ): Array<FloatArray> {
    fun classifySequence (): Array<FloatArray> {

        val interpreter = Interpreter( loadModelFile() )

        val inputs : Array<FloatArray> = arrayOf(
            floatArrayOf( 0.3f, 0.8f, 0.4f, 0.5f ),
            floatArrayOf( 0.4f, 0.1f, 0.8f, 0.5f ),
            floatArrayOf( 0.7f, 0.9f, 0.8f, 0.4f),
        )


        // 3-classes probabilities in the output
        val outputs : Array<FloatArray> = arrayOf(
            floatArrayOf( 0.0f,0.0f,0.0f ),
            floatArrayOf( 0.0f,0.0f,0.0f ),
            floatArrayOf( 0.0f,0.0f,0.0f )
        )
        interpreter.run( inputs , outputs )

        Log.d("ModelOutputs",
            "example 0: "
                    + outputs[0][0].toString()
                    + " - "
                    + outputs[0][1].toString()
                    + " - "
                    + outputs[0][2].toString()
        )
        Log.d("ModelOutputs",
            "example 1: "
                    +
                    outputs[1][0].toString()
                    + " - "
                    + outputs[1][1].toString()
                    + " - "
                    + outputs[1][2].toString()
        )
        Log.d("ModelOutputs",
            "example 2: "
                    +
                    outputs[2][0].toString()
                    + " - "
                    + outputs[2][1].toString()
                    + " - "
                    + outputs[2][2].toString()
        )
        return outputs
    }
}
package com.example.aula1

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.CompoundButton
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.example.aula1.databinding.ActivityMainBinding

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
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

/*var imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .build()*/

var imageProcessorBuilder: ImageProcessor.Builder? = null
var imageProcessor: ImageProcessor? = null
var bitmap: Bitmap? = null

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
var tensorImage: TensorImage? = TensorImage(DataType.UINT8)
var processedBitmap: Bitmap? = null




// Toggle buttons checked state change
fun opToggleButtonChangeListeners(bitmap: Bitmap, previewBitmap: ImageView):
        CompoundButton.OnCheckedChangeListener {
    return CompoundButton.OnCheckedChangeListener { buttonView, isChecked ->

        binding.previewBitmap.setImageBitmap(bitmap)
        //Log.d("MyTag", buttonView.tag as String + " " + isChecked)
        opMap[buttonView.tag as String] = isChecked
        tensorImage = TensorImage(DataType.UINT8)
        tensorImage?.load(bitmap)

        for ((key, value) in opMap) {
            if (value) {
                try {
                    imageProcessorBuilder = ImageProcessor.Builder()
                    imageProcessorBuilder?.add(opMapF[key] as ImageOperator)
                    imageProcessor = imageProcessorBuilder?.build()
                    tensorImage = imageProcessor?.process(tensorImage)
                } catch (e: ClassCastException) {
                    // Used for normalize operator
                    imageProcessorBuilder = ImageProcessor.Builder()
                    imageProcessorBuilder?.add(opMapF[key] as TensorOperator)
                    imageProcessor = imageProcessorBuilder?.build()
                    tensorImage = imageProcessor?.process(tensorImage)
                }

            }
        }
        // Processed TensorImage back into an bitmap
        processedBitmap = tensorImage?.bitmap
        binding.previewBitmap.setImageBitmap(processedBitmap)
    }
}


// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.


private lateinit var binding: ActivityMainBinding

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //Ref: https://developer.android.com/topic/libraries/view-binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        // Sample image loader
        var bitmap = assets
            .open("cat1.png")
            .use(BitmapFactory::decodeStream)

        binding.previewBitmap.setImageBitmap(bitmap)

        // Loading sampling image into a TensorImage
        tensorImage?.load(bitmap)
        tensorImage = imageProcessor?.process(tensorImage)
        val results = classifySequence()

        // Processed TensorImage back into an bitmap
        processedBitmap = tensorImage?.bitmap

        binding.resToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.cropToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.padToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.rotToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.grayToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.normToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
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

    //fun classifySequence ( sequence : IntArray ): FloatArray {
    fun classifySequence (): FloatArray {
        val interpreter = Interpreter( loadModelFile() )
        //val inputs : Array<FloatArray> = arrayOf( sequence.map{ it.toFloat() }.toFloatArray() )
        val inputs : Array<FloatArray> = arrayOf(
            floatArrayOf( 0.3f, 0.8f, 0.4f, 0.5f ),
            floatArrayOf( 0.4f, 0.1f, 0.8f, 0.5f ),
            floatArrayOf( 0.7f, 0.9f, 0.8f, 0.4f ),
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
                    + outputs[1][0].toString()
                    + " - "
                    + outputs[2][0].toString()
        )
        Log.d("ModelOutputs",
            "example 1: "
                    +
                    outputs[0][1].toString()
                    + " - "
                    + outputs[1][1].toString()
                    + " - "
                    + outputs[2][1].toString()
        )
        Log.d("ModelOutputs",
            "example 2: "
                    +
                    outputs[0][2].toString()
                    + " - "
                    + outputs[1][2].toString()
                    + " - "
                    + outputs[2][2].toString()
        )
        return outputs[0]
    }
}
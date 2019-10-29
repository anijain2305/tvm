# from keras.preprocessing import image
import tvm.relay as relay
import tvm
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime as debug_runtime
import numpy as np
import tflite.Model
import time
import glob
from tvm.relay.frontend.tensorflow_parser import TFParser
import argparse

try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

is_profiler = 0
quick_accuracy = 0
check_perf = 0
## DONT DELETE FOR NOW
                # img = image.load_img(image_instance, target_size=(input_shape[1], input_shape[2]))
                # x = image.img_to_array(img, dtype=dtype)
                # x = np.expand_dims(x, axis=0)
                # if dtype == "float32":
                #     x[:, :, :, 0] = 2.0 / 255.0 * x[:, :, :, 0] - 1
                #     x[:, :, :, 1] = 2.0 / 255.0 * x[:, :, :, 1] - 1
                #     x[:, :, :, 2] = 2.0 / 255.0 * x[:, :, :, 2] - 1

##############################
# Define Input Infos
##############################
class InputInfo(object):
    def __init__(self, input_name, input_shape, dtype):
        self.name = input_name
        self.shape = input_shape
        self.dtype = dtype


input_infos = {
    'inception_v3.tflite' : InputInfo('input', (1, 299, 299, 3), 'float32'),
    'inception_v4.tflite' : InputInfo('input', (1, 299, 299, 3), 'float32'),
    'mobilenet_v1_1.0_224.tflite' :  InputInfo('input', (1, 224, 224, 3), 'float32'),
    'mobilenet_v2_1.0_224.tflite' :  InputInfo('input', (1, 224, 224, 3), 'float32'),
    'alexa1' : InputInfo('input', (76, 64), 'float32'),
    # 'inception_v4_299_quant.tflite' : InputInfo('input', (1, 299, 299, 3), 'uint8'),
    # 'inception_v1_224_quant.tflite' : InputInfo('input', (1, 224, 224, 3), 'uint8'),
    # 'inception_v2_224_quant.tflite' : InputInfo('input', (1, 224, 224, 3), 'uint8'),
    # 'mobilenet_v1_0.25_128_quant.tflite' : InputInfo('input', (1, 224, 224, 3), 'uint8'),
    # 'densenet.tflite' : InputInfo('input', (1, 224, 224, 3), 'float32'),
    'resnet_v2_fp32_savedmodel_NHWC' : InputInfo('input', (64, 224, 224, 3), 'float32'),

    'mobilenet_v1_0.25_128_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.25_160_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.25_192_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.25_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.5_128_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.5_160_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.5_192_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.5_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.75_128_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.75_160_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.75_192_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_0.75_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_1.0_128_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_1.0_160_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_1.0_192_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v1_1.0_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'mobilenet_v2_1.0_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'inception_v1_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'inception_v2_224_quant.tflite' :  InputInfo('input', (1, 224, 224, 3), 'uint8'),
    'inception_v3_quant.tflite' : InputInfo('input', (1, 299, 299, 3), 'uint8'),
    'inception_v4_299_quant.tflite' : InputInfo('input', (1, 299, 299, 3), 'uint8'),

}


##############################
# Define Preprocessing
##############################
class TFInceptionPreProcessing(object):
    def crop_center(self, img, threshold=0.875):
        y,x,z = img.shape
        startx = int((x - x * threshold) / 2)
        starty = int((y - y * threshold) / 2)
        x_size = x - startx * 2
        y_size = y - starty * 2
        return img[starty:starty+y_size,startx:startx+x_size,:]

    def preprocessing(self, image_instance, input_info):
        import cv2
        mean_rgb = [123.68, 116.779, 103.939]
        im = cv2.imread(image_instance)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = self.crop_center(im)
        im = cv2.resize(im, dsize=(input_info.shape[1], input_info.shape[2]))
        im = np.expand_dims(im, axis=0)
        # im = im - np.array(mean_rgb)
        # im = im/256
        # im = np.subtract(im, 0.5)
        # im = np.multiply(im, 2.0)
        im = im.astype(input_info.dtype)
        return im

preprocessors = {
    'inception_v3.tflite' : TFInceptionPreProcessing(),
    'inception_v4.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_1.0_224.tflite' :  TFInceptionPreProcessing(),
    'mobilenet_v2_1.0_224.tflite' :  TFInceptionPreProcessing(),
    # 'inception_v3_quant.tflite' : TFInceptionPreProcessing(),
    # 'inception_v4_299_quant.tflite' : TFInceptionPreProcessing(),
    # 'inception_v1_224_quant.tflite' : TFInceptionPreProcessing(),
    # 'inception_v2_224_quant.tflite' : TFInceptionPreProcessing(),
    # 'mobilenet_v1_0.25_128_quant.tflite' : TFInceptionPreProcessing(),
    # 'resnet_v2_fp32_savedmodel_NHWC' : TFInceptionPreProcessing(),
    'alexa1' : TFInceptionPreProcessing(),


    'mobilenet_v1_0.25_128_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.25_160_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.25_192_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.25_224_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.5_128_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.5_160_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.5_192_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.5_224_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.75_128_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.75_160_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.75_192_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_0.75_224_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_1.0_128_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_1.0_160_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_1.0_192_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v1_1.0_224_quant.tflite' : TFInceptionPreProcessing(),
    'mobilenet_v2_1.0_224_quant.tflite' : TFInceptionPreProcessing(),
    'inception_v1_224_quant.tflite' : TFInceptionPreProcessing(),
    'inception_v2_224_quant.tflite' : TFInceptionPreProcessing(),
    'inception_v3_quant.tflite' : TFInceptionPreProcessing(),
    'inception_v4_299_quant.tflite' : TFInceptionPreProcessing(),


}


##############################
# Define Postprocessing
##############################
class PostProcessing(object):
    def postprocessing(self, tensor):
        if (tensor.shape[1] == 1001):
            tensor = tensor[:, 1:]
        predictions = np.squeeze(tensor)
        return predictions

postprocessors = {
     'inception_v3.tflite' : PostProcessing(),
     'inception_v4.tflite' : PostProcessing(),
     'mobilenet_v1_1.0_224.tflite' :  PostProcessing(),
     'mobilenet_v2_1.0_224.tflite' :  PostProcessing(),
     'alexa1' : PostProcessing(),
#     'inception_v3_quant.tflite' : PostProcessing(),
#     'inception_v4_299_quant.tflite' : PostProcessing(),
#     'inception_v1_224_quant.tflite' : PostProcessing(),
#     'inception_v2_224_quant.tflite' : PostProcessing(),
#     'mobilenet_v1_0.25_128_quant.tflite' : PostProcessing(),
    'resnet_v2_fp32_savedmodel_NHWC' : PostProcessing(),


    'mobilenet_v1_0.25_128_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.25_160_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.25_192_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.25_224_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.5_128_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.5_160_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.5_192_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.5_224_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.75_128_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.75_160_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.75_192_quant.tflite' : PostProcessing(),
    'mobilenet_v1_0.75_224_quant.tflite' : PostProcessing(),
    'mobilenet_v1_1.0_128_quant.tflite' : PostProcessing(),
    'mobilenet_v1_1.0_160_quant.tflite' : PostProcessing(),
    'mobilenet_v1_1.0_192_quant.tflite' : PostProcessing(),
    'mobilenet_v1_1.0_224_quant.tflite' : PostProcessing(),
    'mobilenet_v2_1.0_224_quant.tflite' : PostProcessing(),
    'inception_v1_224_quant.tflite' : PostProcessing(),
    'inception_v2_224_quant.tflite' : PostProcessing(),
    'inception_v3_quant.tflite' : PostProcessing(),
    'inception_v4_299_quant.tflite' : PostProcessing(),


}


###################
# Comiler Jobs dispatcher
####################
class CompileDispatcher(object):
    def __init__(self, model_name, target):
        self.model_name = model_name
        self.model_dir = model_name.replace('.tflite', '')
        self.model_file = '/home/ubuntu/tflite_hosted_models/' + self.model_dir + '/' + model_name
        self.input_info = input_infos[model_name]
        self.target = target
        self.current_milli_time = lambda: int(round(time.time() * 1000))

    def parse(self):
        t1 = self.current_milli_time()
        if 'tflite' in self.model_name:
            tflite_model_buf = open(self.model_file, "rb").read()
            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
            mod, params = relay.frontend.from_tflite(
                    tflite_model,
                    shape_dict={self.input_info.name : self.input_info.shape},
                    dtype_dict={self.input_info.name : self.input_info.dtype})
        else: # Tensorflow model
            self.model_file = self.model_file + ".pb"
            parser = TFParser(self.model_file)
            graph_def = parser.parse()
            mod, params = relay.frontend.from_tensorflow(graph_def,
                    shape={self.input_info.name : self.input_info.shape})

        fw = open(self.model_file + '_qnn.txt', 'w')
        fw.write(mod.astext(show_meta_data=False))
        fw.close()
        self.parsed_module = mod
        self.parsed_params = params
        t2 = self.current_milli_time()
        print("Parsing took {} ms".format(t2 - t1))

    def compile(self, opt_level):
        t1 = self.current_milli_time()
        with tvm.target.create(self.target):
            self.parsed_module = relay.qnn.transform.Legalize()(self.parsed_module)
            self.parsed_module = relay.qnn.transform.CanonicalizeOps()(self.parsed_module)
            fw = open(self.model_file + '_relay.txt', 'w')
            fw.write(self.parsed_module.astext(show_meta_data=False))
            fw.close()
        with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build(self.parsed_module,
                                             self.target,
                                             params=self.parsed_params)
        self.compiled_graph = graph
        self.compiled_lib = lib
        self.compiled_params = params
        t2 = self.current_milli_time()
        print("Compilation took {} ms".format(t2 - t1))

    def save(self):
        t1 = self.current_milli_time()
        src_dir = '/home/ubuntu/tflite_compiler_models/'
        path_common = src_dir + self.model_name + '_'
        path_lib = path_common + 'deploy_lib.tar'
        path_graph = path_common + 'deploy_graph.json'
        path_params = path_common + 'depoly_params.params'
        self.compiled_lib.export_library(path_lib)
        with open(path_graph, "w") as fo:
            fo.write(self.compiled_graph)
        with open(path_params, "wb") as fo:
            fo.write(relay.save_param_dict(self.compiled_params))
        t2 = self.current_milli_time()
        print("Saving to file took {} ms".format(t2 - t1))


###################
# Runtime Executor
####################
class RuntimeExecutor(object):
    def __init__(self, model_name, target):
        self.model_name = model_name
        self.target = target
        self.input_info = input_infos[model_name]
        self.preprocessing = preprocessors[model_name].preprocessing
        self.postprocessing = postprocessors[model_name].postprocessing
        self.current_milli_time = lambda: int(round(time.time() * 1000))
        self.images_per_category = 1000
        global quick_accuracy
        if quick_accuracy == 1:
            self.images_per_category = 5


    def load(self):
        global is_profiler
        t1 = self.current_milli_time()
        src_dir = '/home/pi/tflite_compiler_models/'
        self.path_common = src_dir + self.model_name + '_'
        path_lib = self.path_common + 'deploy_lib.tar'
        path_graph = self.path_common + 'deploy_graph.json'
        path_params = self.path_common + 'depoly_params.params'

        graph = open(path_graph).read()
        lib = tvm.module.load(path_lib)
        self.compiled_params = bytearray(open(path_params, "rb").read())

        if is_profiler:
            self.module = debug_runtime.create(graph, lib, tvm.cpu())
        else:
            self.module = runtime.create(graph, lib, tvm.cpu())

        t2 = self.current_milli_time()
        print("Loading the model took {} ms".format(t2 - t1))

    def get_perf(self):

        for i in range(0, 10):
            self.module.run()

        num_iterations = 1000
        total = 0
        for i in range(0, num_iterations):
            t1 = self.current_milli_time()
            self.module.run()
            t2 = self.current_milli_time()
            total += t2 - t1
        latency = (total)/num_iterations
        print("Perf, " + self.model_name + ", " + str(latency))

    def get_profile(self):
        self.module.run()

    def run_imagenet(self):
        global is_profiler
        image_path = '/home/ubuntu/imagenet/val/'
        all_class_path = sorted(glob.glob(image_path+'*'))
        total = 0
        top1_score = 0
        top5_score = 0
        label = 0
        fw = open(self.path_common + 'accuracy_' + self.target.split('=')[-1] + '.txt', 'w')

        for cur_class in all_class_path:
            all_image = glob.glob(cur_class+'/*')
            num_images = 0
            for image_instance in all_image:
                total = total + 1
                num_images = num_images + 1

                # Preprocess the image
                preprocessed_image = self.preprocessing(image_instance, self.input_info)

                # Set the new inputs
                self.module.set_input(self.input_info.name, tvm.nd.array(preprocessed_image))
                self.module.load_params(self.compiled_params)
                self.module.run()
                out_arr = self.module.get_output(0).asnumpy()

                # PostProcess
                predictions = self.postprocessing(out_arr)

                # Get the labels
                labels_sorted = predictions.argsort()[-5:][::-1]
                # print(labels_sorted)

                # Collect statistics
                if labels_sorted[0] == label:
                    top1_score = top1_score + 1
                if label in labels_sorted:
                    top5_score = top5_score + 1
                if not total % 1000:
                    fw.write(str(total) + ',' + str(top1_score/total) + ',' + str(top5_score/total)
                            + '\n')

                if (num_images == self.images_per_category):
                    break
                if is_profiler:
                    break
            label = label + 1
            if is_profiler:
                break


###############
# Helper functions - Refactor
#############
# target = 'llvm -mcpu=cascadelake'
target = 'llvm -device=arm_cpu -model=S2L99 -target=armv7l-linux-gnueabihf -mattr=+neon'
def compile_models(models):
    for model_name in models:
        dispatcher = CompileDispatcher(model_name=model_name,
                                       target=target)

        # Parse the model
        dispatcher.parse()

        # Compile the model
        dispatcher.compile(opt_level=3)

        # Write to disk
        dispatcher.save()

def execute_models(models):
    for model_name in models:
        executor = RuntimeExecutor(model_name=model_name,
                                   target=target)

        # Load the saved model in memory
        executor.load()

        # Load the saved model in memory

        global check_perf
        global is_profiler
        if check_perf == 1:
            executor.get_perf()
        elif is_profiler == 1:
            executor.get_profile()
        else:
            assert False, "Choose perf or profile"
            executor.run_imagenet()


##################################
####### TFLITE codebase
##################################

def run_tflite_graph(tflite_model_buf, input_data):
    def convert_to_list(x):
        if not isinstance(x, list):
            x = [x]
        return x


    """ Generic function to execute TFLite """
    input_data = convert_to_list(input_data)

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    # Run TFLite graph
    current_milli_time = lambda: int(round(time.time() * 1000))
    t1 = current_milli_time()
    iterations = 5
    for i in range(0, iterations):
        interpreter.invoke()
    t2 = current_milli_time()
    print("TFLite runtime = ", (t2 - t1) / iterations, " ms")

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]['index']))

    return tflite_output

def run_tflite_graphs(models):
    for model_name in models:
        model_dir = model_name.replace('.tflite', '')
        model_file = '/home/ubuntu/tflite_hosted_models/' + model_dir + '/' + model_name
        with open(model_file, "rb") as f:
            tflite_model_buf = f.read()
        data = np.random.uniform(size=(1, 224, 224, 3)).astype('uint8')
        tflite_output = run_tflite_graph(tflite_model_buf, data)


######################################
######## TFLITE codebase ends
#####################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-profiler", type=int, help="0/1, if 1 it profiles the code")
    parser.add_argument("-quick_accuracy", type=int, help="0/1, if 1 it gives accuracy on small ataset")
    parser.add_argument("-check_perf", type=int, help="0/1, gives performance")
    parser.add_argument("-network", type=str, help="network")
    args = parser.parse_args()
    is_profiler = args.profiler
    quick_accuray = args.quick_accuracy
    check_perf = args.check_perf
    network = args.network

    # my_models =  ['inception_v1_224_quant.tflite']
    # my_models =  ['inception_v2_224_quant.tflite']
    # my_models =  ['inception_v3_quant.tflite']
    # my_models =  ['inception_v4_299_quant.tflite']
    # my_models =  ['mobilenet_v1_0.25_128_quant.tflite']
    # my_models =  ['resnet_v2_fp32_savedmodel_NHWC']

    if network:
        my_models = [network]
    else:
        my_models = [
        # 'mobilenet_v1_0.25_128_quant.tflite',
        # 'mobilenet_v1_0.25_160_quant.tflite',
        # 'mobilenet_v1_0.25_192_quant.tflite',
        # 'mobilenet_v1_0.25_224_quant.tflite',
        # 'mobilenet_v1_0.5_128_quant.tflite',
        # 'mobilenet_v1_0.5_160_quant.tflite',
        # 'mobilenet_v1_0.5_192_quant.tflite',
        # 'mobilenet_v1_0.5_224_quant.tflite',
        # 'mobilenet_v1_0.75_128_quant.tflite',
        # 'mobilenet_v1_0.75_160_quant.tflite',
        # 'mobilenet_v1_0.75_192_quant.tflite',
        # 'mobilenet_v1_0.75_224_quant.tflite',
        # 'mobilenet_v1_1.0_128_quant.tflite',
        # 'mobilenet_v1_1.0_160_quant.tflite',
        # 'mobilenet_v1_1.0_192_quant.tflite',

        # 'mobilenet_v1_1.0_224_quant.tflite',
        # 'mobilenet_v2_1.0_224_quant.tflite',
        # 'inception_v1_224_quant.tflite',
        # 'inception_v2_224_quant.tflite',
        # 'inception_v3_quant.tflite',
        # 'inception_v4_299_quant.tflite',


        # 'inception_v3.tflite',
        # 'inception_v4.tflite',
        # 'mobilenet_v1_1.0_224.tflite',
        # 'mobilenet_v2_1.0_224.tflite',
        ]
        my_models = ["alexa1"]


    # compile_models(my_models)
    execute_models(my_models)
    # run_tflite_graphs(my_models)

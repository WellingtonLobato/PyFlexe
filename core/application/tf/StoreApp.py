import core.proto.flexe_pb2 as flexe_pb2
import core.proto.flexe_pb2_grpc as flexe_pb2_grpc
import math
import os
import gc

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from core.server.parameter import ndarrays_to_parameters, parameters_to_ndarrays, bytes_to_ndarray, ndarray_to_bytes
from core.server.aggregate import aggregate, aggregate_asynchronous, aggregate_asynchronous_alpha
from core.dataset.dataset_utils_tf import ManageDatasets
from glob import glob


global model_path, initial_path
model_path = ""
initial_path = ""

class StoreApp(flexe_pb2_grpc.FlexeServicer):
    def __init__(self, n_clients, dataset, non_iid):
        self.dataset = dataset
        self.n_clients = n_clients
        self.non_iid = non_iid
        if non_iid == False:
            self.x_train, self.y_train, self.x_test, self.y_test = ManageDatasets(0).select_dataset(self.dataset, self.n_clients, self.non_iid)
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = ManageDatasets(0).select_dataset(self.dataset, self.n_clients, False) 
               
    #Server Functions 
    def server_evaluate(self, request, context):      
        return flexe_pb2.EvaluateReply(loss=int(7), accuracy=int(77))

    def aggregate_evaluate(self, request, context):
        global model_path
                
        print("SERVER - AGGREGATE_EVALUATE datasetPath: ", 
        request.datasetPath, " modelName: ", 
        request.modelName, " epochs: ", 
        request.epochs, " batch_size: ", request.batch_size)

        loss = -1
        accuracy = -1

        # LOAD MODEL
        fileName = model_path
        isExist = os.path.exists(fileName)
        if isExist:
            model = tf.keras.models.load_model(fileName)
            data = model.evaluate(self.x_test, self.y_test, batch_size=int(request.batch_size))
            loss = float(data[0])
            accuracy = float(data[1])
            history={}
            history["loss"] = [loss]
            history["accuracy"] = [accuracy]
            with open(fileName.replace("h5", "txt"), 'a') as file_save:
                file_save.write("AGGREGATE_EVALUATE: "+str(history)+"\n")
        else:
            print("AGGREGATE_EVALUATE VOID_MODEL")
            model = []
        # LOAD MODEL

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        
        return flexe_pb2.EvaluateReply(loss=loss, accuracy=accuracy)


    def aggregate_fit(self, request, context):       
        global model_path
        request.idVehicle = request.idVehicle - 1        
        print("SERVER ", (request.idVehicle), 
        " - AGGREGATE_FIT request.number_of_vehicles:", 
        request.number_of_vehicles, " - AGGREGATE_FIT request.num_examples: ", 
        request.num_examples, "\n")

        list_models = []
        modelName = model_path
        
        # LOAD MODEL
        fileName = modelName.replace("aggregate.h5", "")+str(request.idVehicle)+".h5"
        model = tf.keras.models.load_model(fileName)
        # LOAD MODEL
        
        # WEIGHTS FROM MODEL
        tensors = model.get_weights()
        list_models.append((tensors, request.num_examples))
        # WEIGHTS FROM MODEL

        # LOAD AGG MODEL 
        fileName = model_path
        model_agg = tf.keras.models.load_model(fileName)
        tensors_agg = model_agg.get_weights()
        list_models.append((tensors_agg, request.num_examples))
        # LOAD AGG MODEL 

        # MAKE AGGREGATION
        weights_aggregated = aggregate_asynchronous(list_models)
        model_agg.set_weights(weights_aggregated)
        model_agg.save(fileName)
        # MAKE AGGREGATION

        data = model_agg.evaluate(self.x_train, self.y_train)
        
        print("AGGREGATE_FIT LOSS: ", data[0])
        print("AGGREGATE_FIT ACC: ", data[1], "\n\n")

        history={}
        history["loss"] = [float(data[0])]
        history["accuracy"] = [float(data[1])]

        with open(fileName.replace("h5", "txt"), 'a') as file_pi:
            file_pi.write("AGGREGATE_FIT: "+str(history)+"\n") 

        # WEIGHTS TO BYTES (FROM FLOWER)
        weights_aggregated = model_agg.get_weights()
        new_tensors = [ndarray_to_bytes(ndarray) for ndarray in weights_aggregated]
        # WEIGHTS TO BYTES (FROM FLOWER)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del weights_aggregated
        del tensors
        del tensors_agg
        del list_models
        del model_agg
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        
        return flexe_pb2.ModelReply(idVehicle=int(-1), tensors=new_tensors)

    def aggregate_sync_fit(self, request, context):
        global model_path
        print("SERVER - AGGREGATE_SEMIASYNC_FIT msg: ", request.msg)
        
        list_models = []
        
        # LOAD AGG MODEL 
        fileName = model_path
        model_agg = tf.keras.models.load_model(fileName)
        tensors_agg = model_agg.get_weights()
        list_models.append((tensors_agg, 50000))
        # LOAD AGG MODEL 
        
        veh_ids = request.msg.split(";")
        for veh_id in veh_ids:
            if veh_id.isdigit():
                fileName = model_path.replace("global.h5", "")+str(int(veh_id) - 1)+"_cache.h5"
                print("fileName CACHE: ", fileName)
                model = tf.keras.models.load_model(fileName)
                tensors = model.get_weights()
                list_models.append((tensors, 50000))
        
        # MAKE AGGREGATION
        weights_aggregated = aggregate(list_models)
        model_agg.set_weights(weights_aggregated)
        model_agg.save(model_path)
        # MAKE AGGREGATION

        data = model_agg.evaluate(self.x_train, self.y_train)

        history={}
        history["loss"] = [float(data[0])]
        history["accuracy"] = [float(data[1])]

        with open(model_path.replace("h5", "txt"), 'a') as file_pi:
            file_pi.write("AGGREGATE_SEMIASYNC_FIT: "+str(history)+"\n") 

        # WEIGHTS TO BYTES (FROM FLOWER)
        weights_aggregated = model_agg.get_weights()
        new_tensors = [ndarray_to_bytes(ndarray) for ndarray in weights_aggregated]
        # WEIGHTS TO BYTES (FROM FLOWER)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del weights_aggregated
        del tensors
        del tensors_agg
        del list_models
        del model_agg
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        
        return flexe_pb2.ModelReply(idVehicle=int(-1), tensors=new_tensors)


    def initialize_parameters(self, request, context):
        global model_path, initial_path
        
        print("SERVER - INITIALIZE_PARAMETERS datasetPath: ", 
        request.datasetPath, " modelName: ", request.modelName, 
        " epochs: ", request.epochs, " batch_size: ", request.batch_size)        

        model_path = request.modelName+"global.h5"
        initial_path = request.modelName+"initial.h5"

        # TRAINING PROCESS
        #CREATE GENERIC MODEL
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        #CREATE GENERIC MODEL

        history = model.fit(self.x_train, self.y_train, epochs=request.epochs, batch_size=int(request.batch_size))
        model.save(model_path)
        model.save(initial_path)

        with open(initial_path.replace("h5", "txt"), 'a') as file_save:
            file_save.write("INITIALIZE_PARAMETERS: "+str(history.history)+"\n")
        # TRAINING PROCESS

        # WEIGHTS TO BYTES (FROM FLOWER)
        weights = model.get_weights()
        tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
        # WEIGHTS TO BYTES (FROM FLOWER)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del weights
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        
        return flexe_pb2.ModelReply(idVehicle=int(-1), tensors=tensors)


    def store_model(self, request, context):
        global model_path
        request.idVehicle = request.idVehicle - 1
        print("SERVER ", (request.idVehicle), 
        " - STORE_MODEL request.number_of_vehicles:", 
        request.number_of_vehicles, 
        " - STORE_MODEL request.num_examples: ", 
        request.num_examples, "\n")
        
    	# BYTES TO WEIGHTS  (FROM FLOWER)
        tensors = [bytes_to_ndarray(bytes) for bytes in request.tensors]
        # BYTES TO WEIGHTS  (FROM FLOWER)
        
        #LOAD GLOBAL MODEL
        model = tf.keras.models.load_model(model_path)
        #LOAD GLOBAL MODEL
        
        # STORE MODEL
        fileName = model_path.replace("global.h5", "")+str(request.idVehicle)+"_cache.h5"
        model.set_weights(tensors)
        model.save(fileName)
        # STORE MODEL
        
        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del tensors
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        
        return flexe_pb2.GenericResponse(reply=int(1))       

    def end(self, request, context):
        global model_path
        print("SERVER - END")
        model_path = model_path.replace("global.h5", "")
        models = glob(model_path+"*.h5")
        for model in models:
            os.remove(model)
        gc.collect() # Force Garbage Collect
        return flexe_pb2.GenericResponse(reply=int(1))       


    #Client Functions 
    def fit(self, request, context):
        global initial_path        
        request.idVehicle = request.idVehicle - 1
        print(request.idVehicle, " - FIT datasetPath: ", 
        request.datasetPath, " modelName: ", request.modelName, 
        " epochs: ", request.epochs, " batch_size: ", request.batch_size)
        

        fileName = str(request.modelName)+str(request.idVehicle)+".h5"
        isExist = os.path.exists(fileName)

        if isExist:
            #LOAD MODEL IF EXIST
            model = tf.keras.models.load_model(fileName)
            #LOAD MODEL IF EXIST
        else:
            #LOAD INITIAL MODEL
            model = tf.keras.models.load_model(initial_path)
            #LOAD INITIAL MODEL

        history = model.fit(self.x_train, self.y_train, epochs=request.epochs, batch_size=int(request.batch_size))
        model.save(fileName)
        
        with open(fileName.replace("h5", "txt"), 'a') as file_save:
            file_save.write("FIT: "+str(history.history)+"\n")

        # WEIGHTS TO BYTES (FROM FLOWER)
        weights = model.get_weights()
        tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
        # WEIGHTS TO BYTES (FROM FLOWER)
  
        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del weights
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY
        return flexe_pb2.ModelReply(idVehicle=int(self.y_train.size), tensors=tensors)



    def evaluate(self, request, context):   
        request.idVehicle = request.idVehicle - 1     
        print(request.idVehicle, " - EVALUATE datasetPath: ", 
        request.datasetPath, " modelName: ", request.modelName, 
        " epochs: ", request.epochs, " batch_size: ", request.batch_size)

        # LOAD MODEL
        fileName = str(request.modelName)+str(request.idVehicle)+".h5"
        model = tf.keras.models.load_model(fileName)
        # LOAD MODEL

        data = model.evaluate(self.x_test, self.y_test)
        history={}
        history["loss"] = [float(data[0])]
        history["accuracy"] = [float(data[1])]
        with open(fileName.replace("h5", "txt"), 'a') as file_save:
            file_save.write("EVALUATE: "+str(history)+"\n")
        
        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY  

        return flexe_pb2.EvaluateReply(loss=float(data[0]), accuracy=float(data[1]))
 

    def update_model(self, request, context):
        global model_path
        request.idVehicle = request.idVehicle - 1        
        print(request.idVehicle, " - UPDATE_MODEL datasetPath: ", 
        request.datasetPath, " modelName: ", request.modelName, 
        " epochs: ", request.epochs, " batch_size: ", request.batch_size)

        fileName = str(request.modelName)+str(request.idVehicle)+".h5"            
        model = tf.keras.models.load_model(model_path)
        model.save(fileName)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        del model
        gc.collect() # Force Garbage Collect
        # CLEAN MEMORY 

        return flexe_pb2.GenericResponse(reply=int(1))   


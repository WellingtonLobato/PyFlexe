from concurrent import futures
import logging
from optparse import OptionParser

import grpc
import core.proto.flexe_pb2_grpc as flexe_pb2_grpc
from core.application.tf.StoreApp import StoreApp

import sys
import os

def serve(ip_server, n_clients, dataset, non_iid):
    print("+-------------------------------+")
    print("|       GRPC Server Start!      |")
    print("+-------------------------------+")
    print("Server IP address: " + ip_server)
    print()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
    options = [
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
        ('grpc.client_idle_timeout_ms', 7200000),
        ('grpc.keepalive_timeout_ms', 7200000),
        ('grpc.grpclb_call_timeout_ms', 7200000)
    ])
    flexe_pb2_grpc.add_FlexeServicer_to_server(StoreApp(n_clients, dataset, non_iid), server)
    
    server.add_insecure_port(ip_server+':5000')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = OptionParser()
    
    parser.add_option("-c", "--clients",     		dest="n_clients",          default=1,        help="Number of clients in the simulation",    metavar="INT")
    parser.add_option("-i", "--ip",    		        dest="ip_address",          default='127.0.0.1',  help="Strategy of the federated learning",  metavar="STR")
    parser.add_option("-s", "--strategy",    		dest="strategy_name",      default='FedSGD',  help="Strategy of the federated learning",     metavar="STR")
    parser.add_option("-a", "--aggregation_method", dest="aggregation_method", default='None',    help="Algorithm used for selecting clients",   metavar="STR")
    parser.add_option("-m", "--model",       		dest="model_name",         default='DNN',     help="Model used for trainning",               metavar="STR")
    parser.add_option("-d", "--dataset",     		dest="dataset",            default='MNIST',   help="Dataset used for trainning",             metavar="STR")
    parser.add_option("",   "--poc",         		dest="poc",                default=0,         help="Percentage clients to be selected",      metavar="FLOAT")
    parser.add_option("",   "--decay",       		dest="decay",              default=0,         help="Decay factor for FedLTA",                metavar="FLOAT")
    parser.add_option("",   "--non-iid",     		dest="non_iid",            default=False,     help="Non IID distribution",                   metavar="BOOLEAN")
    parser.add_option("-y", "--classes",     		dest="n_classes",          default=10,        help="Number of classes",                      metavar="INT")
    parser.add_option("-t", "--type",               dest="type",               default='tf',      help="Neural network framework (tf or torch)", metavar="STR")
    
    (opt, args) = parser.parse_args()
    logging.basicConfig()
    serve(opt.ip_address, opt.n_clients, opt.dataset, opt.non_iid)

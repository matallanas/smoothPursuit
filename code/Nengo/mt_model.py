#!/usr/bin/env python

import numpy as np
import scipy.io as io

import nengo
import pyopencl as cl
from nengo_ocl import sim_ocl
from nengo.utils.numpy import rmse, norm
import matplotlib.pyplot as plt
from nengo.utils.logging import logger

class _InputImage(object):
  """Structure of the input image to the mode
     
    Parameters
    ----------
    height: height of the image
    width: width of the image
    height_k: height of the kernel image
    width_k: width of the kernel image
  """
  def __init__(self,height,width,height_k,width_k):
    self.height = height
    self.width = width
    self.height_k = height_k
    self.width_k = width_k

img = _InputImage(240,320,4,4)

def OclSimulator(network):
  ctx = cl.create_some_context()
  return sim_ocl.Simulator(network, context=ctx)

def get_directions():
  verticalPieces = img.height / img.height_k
  horizontalPieces = img.width / img.width_k
  directions = []
  for i in range(0, verticalPieces): 
    for j in range(0, horizontalPieces):
      iny = np.array(range(j*img.width_k,(j+1)*img.width_k)) 
      inxy = iny*img.height + i*img.height_k 
      idx=np.array([],dtype = int)
      for z in range(0, img.height_k):
        vix = inxy + z*np.ones(img.width_k, dtype=np.int)
        idx = np.append(idx,vix)
		
      temp_dirs = np.zeros(img.height*img.width, dtype = np.int)
      temp_dirs[idx] = 1   
      directions.append(temp_dirs)
  
  return directions

def mt_model(Simulator, nl):
  mat = io.loadmat('/home/matallanas/Documents/smoothPursuit/LKPYR/flow-vector.mat')
  speed = mat['Vx']
  s2 = speed[0:240,0:320]
  s2 = np.reshape(s2,240*320,1)
  l  = s2.shape 
  print l
  print speed
  #"""A network that represents sin(t)."""
  N = 768000
  mt = nengo.Network(label='mt_model')
  with mt:
    input = nengo.Node(output=s2, dimensions=76800)
    mt_neurons = nengo.Ensemble(nl(N), radius=20, dimensions=76800)
    nengo.Connection(input, mt_neurons)
    in_p = nengo.Probe(input, 'output')
    mt_p = nengo.Probe(mt_neurons, 'decoded_output', synapse=0.02)

  sim = Simulator(mt)
  sim.run(5.0)
  print sim.data[mt_p] 

    #t = sim.trange()
    #plt.plot(t, sim.data[in_p], label='Input')
    #plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.02')
    #plt.legend(loc=0)
    #plt.show()

    #target = np.sin(np.arange(5000) / 1000.)
    #target.shape = (-1, 1)
    #logger.debug("[New API] input RMSE: %f", rmse(target, sim.data[in_p]))
    #logger.debug("[New API] A RMSE: %f", rmse(target, sim.data[A_p]))
    #assert rmse(target, sim.data[in_p]) < 0.001
    #assert rmse(target, sim.data[A_p]) < 0.1
mt_model(OclSimulator,nengo.LIF)

#encoders = get_directions()
#print encoders[0]
#print np.nonzero(encoders[0]>0.5)
#print np.nonzero(encoders[1]>0.5)

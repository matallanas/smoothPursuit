import matplotlib.pyplot as plt
import nengo
import nengo.helpers
import numpy as np
from Delayy import Del

import pyopencl as cl
from nengo_ocl.sim_ocl import Simulator
from nengo.tests.helpers import Plotter

import csv
import time
import socket
import struct

ctx = cl.create_some_context()

UDP_IP = "129.97.172.65"
MY_IP = "129.97.172.66"
#UDP_IP = "127.0.0.1"
UDP_PORT = 4012
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
sock.bind((MY_IP, UDP_PORT))
simtime = .001
  
data = 0,0
indx=0

D1 = Del(72-25)
D2 = Del(5)

model = nengo.Model(label='SPModel')

with model:
    frq = 1
    
    #data = 0
    #indx = 0
    kp = 5.06
    kd = 0.105
    #indx =indx+1
	
    #input = nengo.Node(output= lambda t: 2*np.sin(2*np.pi*frq*t))
    #input = nengo.Node(output= 15)
  	
    def receive (t):    	
    	#global data    	 
    	#ind = int(1000*t)
    	global indx 
    	x = data[indx]
    	indx = indx +1
    	
    	
    	#print 'input',x ,'indx',indx   	  	
    	#print 'time', t,'input', x, 'ii',ii
    	return x
    	
    
    input = nengo.Node(receive)	
    #input = nengo.Node(output= lambda t: data [int(1000*t)])	
    
    def send(t, x):
    	if ((1000*simtime-indx-1) < .01):
    		#print 'time', t, 'output', x, 'indx',indx , 'mard', 1000*simtime-indx
    		x_net = struct.pack('!1d' , *x)
    		#print 'simtime', simtime
    		sock.sendto(x_net, (UDP_IP, UDP_PORT))
        
    output = nengo.Node(send, dimensions=1)
    
    
    def dd1 (t,x):
    	return D1.step(x)
    	 
    delay_MT_MST = nengo.Node(dd1,dimensions=1)
      
    '''
    MT = nengo.Ensemble(nengo.Direct(), dimensions=1 , radius = 20)
    u = nengo.Ensemble(nengo.Direct(), dimensions=1,radius = 20)
    f = nengo.Ensemble(nengo.Direct(), dimensions=1,radius = 70)

    x1 = nengo.Ensemble(nengo.Direct(), dimensions=1,radius = 40)
    x2 = nengo.Ensemble(nengo.Direct(), dimensions=1,radius = 90)
    Intg = nengo.Ensemble(nengo.Direct(), dimensions=1,radius = 20)
    '''
    MT = nengo.Ensemble(nengo.LIF(1000), dimensions=1 , radius = 20)
    u = nengo.Ensemble(nengo.LIF(1000), dimensions=1,radius = 20)
    f = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 70)

    x1 = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 40)
    x2 = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 90)
    Intg = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 20)
	
	#'''
    def dd2 (t,x):
	   	return D2.step(x)
	   	
    delay_FEF_DLPN = nengo.Node(dd2,dimensions=1)
   
    Eye = nengo.Ensemble(nengo.Direct(), dimensions=1)
       
   
    tau = 0.01

   
    nengo.Connection(input, MT,transform=[[1]])
    nengo.Connection(Eye, MT,transform=[[-1]],filter=.015)
   
    nengo.Connection(MT,delay_MT_MST)  

    nengo.Connection(delay_MT_MST, u,filter=.005)
       
    nengo.Connection(u, f, transform=[[tau/.055]], function=lambda x: kp*(x[0]),filter=tau)
    nengo.Connection(f, f, transform=[[1-(tau/.055)]], filter=tau)
    
   
    #### FEF

    omg = 250
  
    pp1 = .03
    pp2 = .01
   
    nengo.Connection(u, x1, transform=[[tau*pp1*omg*omg]], function=lambda x: 40.0876*(2./(1+np.exp(-0.2174*x[0]))-1), filter=tau)

    nengo.Connection(x1, x1, transform=[[1-tau*omg]], filter=tau)
    nengo.Connection(x2, x1, transform=[[tau*pp1*omg/pp2]], filter=tau)
   
   
    nengo.Connection(u, x2, transform=[[-tau*pp2*omg*omg]], function=lambda x: 40.0876*(2./(1+np.exp(-0.2174*x[0]))-1), filter=tau)

    nengo.Connection(x2, x2, transform=[[1-tau*omg]], filter=tau)
   
    nengo.Connection(x1, delay_FEF_DLPN, transform=[[1/pp1]], function=lambda x: kd*(x[0]))

    ####DLPN
       
    nengo.Connection(f, Intg, transform=[[0.1]], filter=0.1)
    nengo.Connection(delay_FEF_DLPN, Intg, transform=[[0.1]],filter=0.1)

    nengo.Connection(Intg, Intg, transform=[[1]], filter=0.1)
    
   
    nengo.Connection(Intg, Eye, transform=[[1]])
    nengo.Connection(Eye, output, transform=[[1]])

	   
    #nengo.Connection(a, sina,  ,filter=tau)   
    p1 = nengo.Probe(input, 'output')
    p2 = nengo.Probe(MT, 'decoded_output', filter = 0.01)
    p3 = nengo.Probe(f, 'decoded_output', filter = 0.01)
    p4 = nengo.Probe(u, 'decoded_output', filter = 0.01)
    p5 = nengo.Probe(x1, 'decoded_output', filter = 0.01)
    p6 = nengo.Probe(x2, 'decoded_output', filter = 0.01)
    p7 = nengo.Probe(Intg, 'decoded_output', filter = 0.01)
    p8 = nengo.Probe(Eye, 'decoded_output', filter = 0.01)


# Create our simulator
sim = nengo.Simulator(model)
#sim = Simulator(model, context=ctx)
# Run it for 6 seconds

i=598

print 'ready'
t_sim=time.time()

data_net, addr = sock.recvfrom(1024)
garb = struct.unpack('!1d', data_net)
print 'garb', garb

while i>0:
	data_net, addr = sock.recvfrom(1024)
	data1 = struct.unpack('!1d', data_net)

	if (data1):
		#print 'sim_data',data1
		simtime = data1[0]
		data_net, addr = sock.recvfrom(1024)
		num= int(1000*simtime)
		data = struct.unpack('!%sd' % num, data_net)
		#print 'dataa',data
		indx=0
		sim.run(data1[0])	
		#print time.time()-t_sim
		i = i -1
		#print 'i',i
		#break

#sim.run(.999)
#elpased_sim = time.time()-t_sim


# Plot the decoded output of the ensemble

t = sim.trange() # Get the time steps

plt.plot(t, sim.data(p1), label="Input")
'''
plt.plot(t, sim.data(p2), label="MT")
plt.plot(t, sim.data(p3), label="f")
plt.plot(t, sim.data(p4), label="u")
plt.plot(t, sim.data(p5), label="x1")
plt.plot(t, sim.data(p6),  label="x2")
plt.plot(t, sim.data(p7),  label="intg")
'''
plt.plot(t, sim.data(p8), 'k', label="eye")
plt.legend()
'''
myfile = open ('Myfile2.csv','wb') 
writer =csv.writer(myfile)	
writer.writerow([[sim.data(p8)]])
'''


np.savetxt('/home/bselby/Nenngoo/networkx-1.8.1/networkx/examp_output.txt', sim.data(p8), delimiter = ',')


plt.show()

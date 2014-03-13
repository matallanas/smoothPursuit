import matplotlib.pyplot as plt
import nengo
import nengo.helpers
import numpy as np
from Delayy import Del

D1 = Del(72)
D2 = Del(5)

model = nengo.Model(label='SPModel')

with model:
    frq = 2
    
    kp = 5.6
    kd = 0.185

    #input = nengo.Node(output= lambda x: 15*np.sin(2*np.pi*frq*x[0]))
    input = nengo.Node(output= 15)
  
    MT = nengo.Ensemble(nengo.LIF(1000), dimensions=1 , radius = 20)
    
    delay_MT_MST = nengo.Node(output = lambda x: D1.step(x[0]))
      
    u = nengo.Ensemble(nengo.LIF(1000), dimensions=1,radius = 20)
    f = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 70)

    x1 = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 40)
    x2 = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 90)
    Intg = nengo.Ensemble(nengo.LIF(2000), dimensions=1,radius = 20)

    delay_FEF_DLPN = nengo.Node(output = lambda x: D2.step(x[0]))
   
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
    
   
    nengo.Connection(Intg, Eye, transform=[[1]], filter=.005)

   
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
# Run it for 6 seconds
sim.run(6)


# Plot the decoded output of the ensemble

t = sim.data(model.t_probe) # Get the time steps
#print t
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

plt.show()
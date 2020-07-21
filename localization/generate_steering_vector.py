import numpy as np
import scipy as sp


#mic_position=[[x,y,z],[x,y,z]]
#source_position=[[x,y,z],[x,y,z] ]
#freqs=[f1,f2,f3...]
def obtain_steering_vector(mic_position,source_position,freqs,SOUND_SPEED=340,useAmp=False):
    mic_num=np.shape(mic_position)[0]
    source_num=np.shape(source_position)[0]


    #mic_position
    #m,x
    #s, x
    #m,s,x
    source_position=np.expand_dims(source_position,axis=0)
    mic_position=np.expand_dims(mic_position,axis=1)
    #m,s
    distance=np.sqrt(np.sum(np.square(source_position-mic_position),axis=2))
    delay=distance/SOUND_SPEED
    #s,freq,m
    phase_delay=np.einsum('k,ms->skm',-2.j*np.pi*freqs,delay)
    #s,f,m
    steering_vector=np.exp(phase_delay)
    if useAmp==True:
        #m,s
        ampRatio=np.divide(1.,distance)
        steering_vector=np.einsum('ms,skm->skm',ampRatio,steering_vector)
    #大きさを1で正規化する
    #skm
    norm=np.sqrt(np.sum(steering_vector*np.conjugate(steering_vector),axis=2,keepdims=True))
    steering_vector=np.divide(steering_vector,norm)

    return(steering_vector)



def obtain_steering_vector_test():
    fftSize=1024
    fftMax=int(fftSize/2+1)
    print(fftMax)
    sampling_rate=16000
    max_freq=sampling_rate/2

    freqs=np.array(range(fftMax),dtype=np.float)/np.array(fftMax,dtype=np.float)*max_freq
    
        

    mic_position=np.array([[0.,0.,0.],[0.05,0.05,0.03],[0.04,0.04,0.02]])
    source_position=np.array([[1.,0.,0.],[0.,1.,0.]])

    #s,k,m
    steeringVector=obtain_steering_vector(mic_position,source_position,freqs,useAmp=True)

    print(np.shape(steeringVector))
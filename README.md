# sigpy_demons
Demons algorithm implemented with sigpy toolbox to enable GPU acceleration.

#Things that need to be done:
1) Test. Test. Test.
2) Optimize for speed. Full Cuda Versions can register datasets (256x256x128) in under 10 seconds. 
Probably not feasible without writing my own kernels, but should approach this limit.

    a) Some improvements could be made with the convolutions. Would ideally implement overlap-add/save depending on the kernel sizes.
    
        aa) Not sure cupy dynamically switches to more efficient convolution methods with kernel sizes.
      
    b) Pyramid downsampling and upsampling could probably be better.
    
    c) Not sure the matrix exponential function is doing what I want... Maps don't look diffeomorphic.
    
    d) Need to check and recheck functions... Jacobian, Determinant, Energy function (cost), etc...
    
    e) Should probably check the filter functions.
    
    f) Warp function should be harmonized to sigpy style coding. Need to make a map_coordinates style function.
    
3) Resample functions are almost definitely not right... 

In short. Need to check everything.

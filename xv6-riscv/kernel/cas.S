.global cas

# a0 holds address of memory location
# a1 holds expected value
# a2 holds desired value
# a0 holds return value, 0 if successful, !0 otherwise

 cas:
     lr.w t0, (a0)         # load original value
     bne t0, a1, fail      # doesn't match' so fail
     sc.w t0, a2, (a0)     # try update
     bnez t0, cas          # retry if store-conditional fail
     li a0, 0              # set return to success
     jr ra                 # return  
 fail: 
     li a0, 1              # set return to failure
     jr ra                 # return
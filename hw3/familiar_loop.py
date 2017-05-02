

class Solution:

	def discount_rewards(r):
	  """ take 1D float array of rewards and compute discounted reward """
	  discounted_r = np.zeros_like(r)
	  running_add = 0
          gamma = 0.99
	  for t in reversed(xrange(0, r.size)):
	    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
	    running_add = running_add * gamma + r[t]
	    discounted_r[t] = running_add
	  return discounted_r


def main():
  r = [5.0,4.0,3.0,2.0,1.0,3.0]
  ans=[]
  sol = Solution() 
  print r
  for t in reversed(xrange(0, r.size)):
      print t  

  ans = sol.discount_rewards(r)      
  print ans

if __name__ == "__main__":
    main()d

import numpy as np
import math
from decimal import Decimal
from jax.typing import ArrayLike

def Compute_real_space_ewald_table(
        nR: int,
        a: float,
        xi: float) -> ArrayLike:

        """Construct the table containing mobility scalar functions values
        as functions of discrete distance. These will be used to linearly interpolate
        obtaining values for any distance. Due to high complexity of the mobility functions,
        calculations are performed in 'extended' precision and then truncate to single precision.
        
        Parameters
        ----------
        nR:
            Number of entries in tabulation, for each scalar mobility function
        a:
            Particle radius
        xi: 
            Ewald splitting parameter
            
            
        Returns
        -------
        ewaldC
    
        """ 
        
        # table discretization in extended precision (80-bit)
        dr_string = "0.00100000000000000000000000000000" # pass value as a string with arbitrary precision
        dr = Decimal(dr_string)                          # convert to float with arbitrary precision
        dr = np.longfloat(dr)                            # convert to numpy long float (truncate to 64/80/128-bit, depending on platform used) 
        
        Imrr = np.zeros(nR)
        rr = np.zeros(nR)
        g1 = np.zeros(nR)
        g2 = np.zeros(nR)
        h1 = np.zeros(nR)
        h2 = np.zeros(nR)
        h3 = np.zeros(nR)

        xxi = np.longfloat(xi)
        a_64 = np.longfloat(a)
        Pi = np.longfloat(np.pi)
        kk = np.arange(nR,dtype=np.longdouble)
        r_array = (kk * dr + dr)
        
        # expression have been simplified assuming no overlap, touching, and overlap

        for i in range(nR):
            
            r = r_array[i]    
            
            if(r>2*a_64):
    
                Imrr[i] = (-math.pow(a_64, -1) + (math.pow(a_64, 2)*math.pow(r, -3))/2. + (3*math.pow(r, -1))/4. + (
                    3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)*(-12*math.pow(r, 4) + math.pow(xxi, -4)))/128
                + math.pow(a_64, -2)*((9*r)/32. -
                                    (3*math.pow(r, -3)*math.pow(xxi, -4))/128.)
                + (math.erfc((2*a_64 + r)*xxi)*(128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) +
                  96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256.
                + (math.erfc(2*a_64*xxi - r*xxi)*(128*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) -
                                                      96*math.pow(r, -1) + math.pow(a_64, -2)*(-36*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/256.
                + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                    * math.pow(r, -2)*math.pow(xxi, -3)*(1 + 6*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3)
                    * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) - 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(2 - 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                    (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) + 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(-2 + 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128.)



                rr[i] =( -math.pow(a_64, -1) - math.pow(a_64, 2)*math.pow(r, -3) + (3*math.pow(r, -1))/2. + (
                            3*math.pow(a_64, -2)*math.pow(r, -3)*(4*math.pow(r, 4) + math.pow(xxi, -4)))/64.
                + (math.erfc(2*a_64*xxi - r*xxi)*(64*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) -
                    96*math.pow(r, -1) + math.pow(a_64, -2)*(-12*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                + (math.erfc((2*a_64 + r)*xxi)*(64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) +
                          96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                          * math.pow(r, -2)*math.pow(xxi, -3)*(-1 + 2*math.pow(r, 2)*math.pow(xxi, 2)))/32.
                - ((2*a_64 + 3*r)*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                            (-1 - 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                + ((2*a_64 - 3*r)*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                            (-1 + 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                - (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)
                          * math.pow(xxi, -4)*(1 + 4*math.pow(r, 4)*math.pow(xxi, 4)))/64.)
                
                g1[i] =    (    (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3) *
                                math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4))) / 640.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                - (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                    a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                            + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
                
                
                g2[i] =  ( (-3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                    * (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                
                
                h1[i] =   (     (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                * (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                  * (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                   * (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
               
                
                h2[i] =  (     (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                * (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                  * (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                    * math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                  * (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                    * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                        
                
                h3[i] =   (     (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                * (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                  * (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                  * (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)
                                                                                                                            * (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                    xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                
                
            elif(r == 2*a_64):
                
                Imrr[i] = -((math.pow(a_64, -5)*(3 + 16*a_64*xxi*math.pow(Pi, -0.5))*math.pow(xxi, -4))/2048. 
                + (3*math.erfc(2*a_64*xxi)*math.pow(a_64, -5) * (-192*math.pow(a_64, 4) + math.pow(xxi, -4)))/1024.
                + math.erfc(4*a_64*xxi)*(math.pow(a_64, -1) -
                                              (3*math.pow(a_64, -5)*math.pow(xxi, -4))/2048.)
                + (math.exp(-16*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                    Pi, -0.5)*math.pow(xxi, -3)*(-1 - 64*math.pow(a_64, 2)*math.pow(xxi, 2)))/256.
                + (3*math.exp(-4*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                    Pi, -0.5)*math.pow(xxi, -3)*(1 + 24*math.pow(a_64, 2)*math.pow(xxi, 2)))/256.)
                
                
                rr[i] = ((math.pow(a_64, -5)*(3 + 16*a_64*xxi*math.pow(Pi, -0.5))*math.pow(xxi, -4))/1024. 
                + math.erfc(
                    2*a_64*xxi)*((-3*math.pow(a_64, -1))/8. - (3*math.pow(a_64, -5)*math.pow(xxi, -4))/512.)
                + math.erfc(4*a_64*xxi)*(math.pow(a_64, -1) +
                                              (3*math.pow(a_64, -5)*math.pow(xxi, -4))/1024.)
                + (math.exp(-16*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                    Pi, -0.5)*math.pow(xxi, -3)*(1 - 32*math.pow(a_64, 2)*math.pow(xxi, 2)))/128.
                + (3*math.exp(-4*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                    Pi, -0.5)*math.pow(xxi, -3)*(-1 + 8*math.pow(a_64, 2)*math.pow(xxi, 2)))/128.)
                
                
                g1[i] = ( (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3) *
                                math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                - (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                    a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                            + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
                
                g2[i] =  (      (-3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                  * (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                
                
                
                h1[i] =   (     (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                                    - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
                         
                
                h2[i] =  (   (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                  * math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                  * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                
                
                h3[i] =   (      (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                  (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2) *
                                                                                                                            (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                    xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                
                
            elif(r < 2*a_64):
                
                Imrr[i] = ( (-9*r*math.pow(a_64, -2))/32 
                + math.pow(a_64, -1) 
                - (math.pow(a_64, 2)*math.pow(r, -3)) /2 
                - (3*math.pow(r, -1))/4 
                + (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)* (-12*math.pow(r, 4)
                + math.pow(xxi, -4)))/128
                + (math.erfc((-2*a_64 + r)*xxi)*(-128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, - 3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256
                + (math.erfc((2*a_64 + r)*xxi)*(128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256
                + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5) * math.pow(r, -2)*math.pow(xxi, -3)*(1 + 6*math.pow(r, 2)*math.pow(xxi, 2)))/64
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) - 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(2 - 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) + 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(-2 + 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128 )
                

                rr[i] = (((2*a_64 + 3*r)*math.pow(a_64, -2) *
                math.pow(2*a_64 - r, 3)*math.pow(r, -3))/16.
                + (math.erfc((-2*a_64 + r)*xxi)*(-64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -
                  3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                + (math.erfc((2*a_64 + r)*xxi)*(64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) +
                  96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                  * math.pow(r, -2)*math.pow(xxi, -3)*(-1 + 2*math.pow(r, 2)*math.pow(xxi, 2)))/32.
                - ((2*a_64 + 3*r)*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                    (-1 - 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                + ((2*a_64 - 3*r)*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                    (-1 + 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                - (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)
                  * math.pow(xxi, -4)*(1 + 4*math.pow(r, 4)*math.pow(xxi, 4)))/64.)

                g1[i] =     ((-9*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6))/128. 
                - (9*math.pow(a_64, -4)*math.pow(r, -2)*math.pow(xxi, -4))/128.
                + (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                              * math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                              (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                              (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                + (3*math.erfc(2*a_64*xxi - r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                        + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
    
                g2[i] =   ((-3*r*math.pow(a_64, -3))/10. - (12*math.pow(a_64, 2)*math.pow(r, -4)) /5. 
                + (3*math.pow(r, -2))/2. + (3*math.pow(a_64, -4)*math.pow(r, 2))/32.
                - (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                  * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                    (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                    (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                  4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                
                
                h1[i] =        ( (9*r*math.pow(a_64, -4))/64. - (3*math.pow(a_64, -3))/10. - (9*math.pow(a_64, 2)*math.pow(
                                    r, -5))/10. + (3*math.pow(r, -3))/4. - (3*math.pow(a_64, -6)*math.pow(r, 3))/512.
                + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                    (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                    (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                    (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                                    - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                            + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
                                
                
                h2[i] =    (   (63*r*math.pow(a_64, -4))/64. - (3*math.pow(a_64, -3))/2. + (9*math.pow(a_64, 2)*math.pow(r, -5))/2. - (3*math.pow(r, -3)
                                                                                                                                  )/4. - (33*math.pow(a_64, -6)*math.pow(r, 3))/512. + (9*math.pow(a_64, -6)*math.pow(r, -3)*math.pow(xxi, -6))/128.
                - (27*math.pow(a_64, -4)*math.pow(r, -3)*math.pow(xxi, -4))/64. + (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                                                                                  (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (3*math.erfc(2*a_64*xxi - r*xxi)*math.pow(a_64, -6)*math.pow(r, -3)*math.pow(xxi, -6)*(-3 + 18*math.pow(a_64, 2)*math.pow(xxi, 2)*(1 - 4*math.pow(r, 4)*math.pow(xxi, 4)) + 128*math.pow(a_64, 6)*math.pow(xxi, 6) + 64*math.pow(a_64, 3)*math.pow(r, 3)*math.pow(xxi, 6)
                                                                                                                            + 8*math.pow(r, 6)*math.pow(xxi, 6)))/256. + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                                                                                                                                                            (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                                                                                                                                                                            - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(
                                                                                                                                                                                a_64, 2)*math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                                                                                                                                                                            + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                    (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                    * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                                    - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(135 + 8*math.pow(xxi, 2)*(-6*(30*math.pow(a_64, 2) + math.pow(r, 2)) + 9*(4*math.pow(a_64, 2) - 3*math.pow(r, 2))*math.pow(r, 2)*math.pow(xxi, 2)
                                                                                                                                                      + 2*(-768*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) + 256*math.pow(a_64, 3)*math.pow(r, 5) - 168*math.pow(a_64, 2)*math.pow(r, 6) + 11*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                            + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                

                h3[i] =       (  (9*r*math.pow(a_64, -4))/64. + (9*math.pow(a_64, 2)*math.pow(r, -5))/2. 
                - (9*math.pow(r, -3))/4. - (9*math.pow(a_64, -6)*math.pow(r, 3))/512.
                + (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                    (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                    (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                    (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2) *
                                                                                                                            (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                      + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                    xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
        
        ewaldC = np.zeros((2*nR, 4))
        ewaldC[0::2, 0]=((Imrr))  # UF1
        ewaldC[0::2, 1]=((rr))  # UF2
        ewaldC[0::2, 2]=((g1/2))  # UC1
        ewaldC[0::2, 3]=((-g2/2))  # UC2
        ewaldC[1::2, 0]=((h1))  # DC1
        ewaldC[1::2, 1]=((h2))  # DC2
        ewaldC[1::2, 2]=((h3))  # DC3
                       
        return ewaldC
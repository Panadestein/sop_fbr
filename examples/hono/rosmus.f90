function potr(r2, r3, r1, t2, t1, t) result(en)
      
      integer i,ii
      integer(8) a,b,c,d,e,f
      real(8) pi,co 
      real(8), intent(in) :: r2,r3,r1,t1,t2,t
      real(4) en
      pi=3.1415927D0
      en=0.0D0

      open(20,file="coeff_hono_pes", status="old")

      do i=1,389
         read(20,*) a,b,c,d,e,f,co,ii

         en=en+(1.0D0-dexp(-0.7D0*(-2.213326D0 + r1)))**a &
         *(1.0D0-dexp(-0.7D0*(-2.696732D0+r2)))**b & 
         *(1.0D0-dexp(-0.7D0*(-1.82291D0+r3)))**c &
         *(-1.931507D0+t1)**d &
         *(-1.777642D0+t2)**e &
         *cos(f*(t-pi))*co   

      enddo

      en=en*dexp(-(-2.2133326D0+r1)**2) & 
      *dexp(-(-2.696732D0+r2)**2/9.0D0) & 
      *dexp(-(-1.82291D0+r3)**2/4.0D0) &
      *dexp(-(-1.931507D0+t1)**2/4.0D0) &
      *dexp(-(-1.777642D0+t2)**2/4.0D0)

      close(20)
      return 

end function potr

program pot_ros      
      
      real(8) :: r1_0,r2_0,r3_0,t1_0,t2_0,t_0
      real(8) :: r1_0_0,r2_0_0,r3_0_0,t1_0_0,t2_0_0,t_0_0
      real(4) :: v, w

!(N=O)
      r1_0_0=2.20092D0
      r1_0=2.21097956679
!(O-N, middle)
      r2_0_0=2.84746D0
      r2_0=2.6947494549
!(OH)
      r3_0_0=1.81801D0
      r3_0=1.82169598494
!(ONO)
      t1_0_0=1.92876D0
      t1_0=1.93207948196
!(HON)
      t2_0_0=1.75775D0
      t2_0=1.77849050778
!(torsion)
      t_0_0=3.1415927D0*86.6D0/180.0D0
      t_0=3.14159265359

v = potr(r2_0, r3_0, r1_0, t2_0, t1_0, t_0)
w = potr(r2_0_0, r3_0_0, r1_0_0, t2_0_0, t1_0_0, t_0_0)
write (*,*) w, v, v - w

end program pot_ros


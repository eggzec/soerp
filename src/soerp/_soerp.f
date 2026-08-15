C-----------------------------------------------------------------------
C     SOERP - Second Order ERror Propagation
C
C     Moment equations of N. D. Cox, "Tolerance Analysis by Computer",
C     Journal of Quality Technology, Vol. 11, No. 2, April 1979,
C     equations (A-6) through (A-9) of the Appendix.
C
C     The output variable is the second-order polynomial
C
C         y = sum_i b_i w_i + sum_i b_ii w_i**2
C             + sum_{i<j} b_ij w_i w_j
C
C     in the standardized, statistically independent inputs w_i, whose
C     central moments are mu_ij (VM(I,J) below, 0 <= J <= 8).
C
C     Naming inside the routines follows the paper:
C         LC(I)   = b_i      standardized linear coefficient
C         QC(I)   = b_ii     standardized pure quadratic coefficient
C         CP(I,J) = b_ij     standardized cross-product coefficient
C         VM(I,J) = mu_ij    Jth standardized central moment of input I
C-----------------------------------------------------------------------

      SUBROUTINE RAWMOM(N, LC, QC, CP, VM, K, ANS)
C     Kth moment of y about the origin, 0 <= K <= 4.
      IMPLICIT NONE
      INTEGER N, K
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8), ANS

      ANS = 0.0D0
      IF (K .EQ. 0) THEN
         ANS = 1.0D0
      ELSE IF (K .EQ. 1) THEN
         CALL RAWM1(N, QC, VM, ANS)
      ELSE IF (K .EQ. 2) THEN
         CALL RAWM2(N, LC, QC, CP, VM, ANS)
      ELSE IF (K .EQ. 3) THEN
         CALL RAWM3(N, LC, QC, CP, VM, ANS)
      ELSE IF (K .EQ. 4) THEN
         CALL RAWM4(N, LC, QC, CP, VM, ANS)
      END IF

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE RAWM1(N, QC, VM, ANS)
C     Equation (A-6).
      IMPLICIT NONE
      INTEGER N, I
      DOUBLE PRECISION QC(N), VM(N,0:8), ANS

      ANS = 0.0D0
      DO 10 I = 1, N
         ANS = ANS + QC(I)*VM(I,2)
   10 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE RAWM2(N, LC, QC, CP, VM, ANS)
C     Equation (A-7).
      IMPLICIT NONE
      INTEGER N, I, J
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8), ANS

      ANS = 0.0D0
      DO 10 I = 1, N
         ANS = ANS + LC(I)**2*VM(I,2)
         ANS = ANS + 2.0D0*LC(I)*QC(I)*VM(I,3)
         ANS = ANS + QC(I)**2*VM(I,4)
   10 CONTINUE

      DO 30 I = 1, N-1
         DO 20 J = I+1, N
            ANS = ANS + (2.0D0*QC(I)*QC(J) + CP(I,J)**2)
     &                  *VM(I,2)*VM(J,2)
   20    CONTINUE
   30 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE RAWM3(N, LC, QC, CP, VM, ANS)
C     Equation (A-8).
      IMPLICIT NONE
      INTEGER N, I, J, M
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8), ANS
      DOUBLE PRECISION T

      ANS = 0.0D0

C     Single-variable terms.
      DO 10 I = 1, N
         ANS = ANS + LC(I)**3*VM(I,3)
         ANS = ANS + QC(I)**3*VM(I,6)
         ANS = ANS + 3.0D0*LC(I)**2*QC(I)*VM(I,4)
         ANS = ANS + 3.0D0*LC(I)*QC(I)**2*VM(I,5)
   10 CONTINUE

C     Pair terms, i < j.
      DO 30 I = 1, N-1
         DO 20 J = I+1, N
            ANS = ANS + CP(I,J)**3*VM(I,3)*VM(J,3)
            ANS = ANS + 6.0D0*LC(I)*LC(J)*CP(I,J)*VM(I,2)*VM(J,2)
            ANS = ANS + 6.0D0*QC(I)*QC(J)*CP(I,J)*VM(I,3)*VM(J,3)
   20    CONTINUE
   30 CONTINUE

C     Cross-diagonal pair terms, j /= i.
      DO 50 I = 1, N
         DO 40 J = 1, N
            IF (J .NE. I) THEN
               ANS = ANS + 3.0D0*QC(I)**2*VM(I,4)*QC(J)*VM(J,2)
               ANS = ANS + 6.0D0*LC(I)*QC(J)*CP(I,J)*VM(I,2)*VM(J,3)
               ANS = ANS + 3.0D0*QC(I)*LC(J)**2*VM(I,2)*VM(J,2)
               ANS = ANS + 6.0D0*LC(I)*QC(I)*QC(J)*VM(I,3)*VM(J,2)
               ANS = ANS + 3.0D0*LC(I)*CP(I,J)**2*VM(I,3)*VM(J,2)
               ANS = ANS + 3.0D0*QC(I)*CP(I,J)**2*VM(I,4)*VM(J,2)
            END IF
   40    CONTINUE
   50 CONTINUE

C     Triplet terms, i < j < m.
      DO 80 I = 1, N-2
         DO 70 J = I+1, N-1
            DO 60 M = J+1, N
               T = 6.0D0*QC(I)*QC(J)*QC(M)
               T = T + 6.0D0*CP(I,J)*CP(I,M)*CP(J,M)
               T = T + 3.0D0*QC(I)*CP(J,M)**2
               T = T + 3.0D0*QC(J)*CP(I,M)**2
               T = T + 3.0D0*QC(M)*CP(I,J)**2
               ANS = ANS + T*VM(I,2)*VM(J,2)*VM(M,2)
   60       CONTINUE
   70    CONTINUE
   80 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE RAWM4(N, LC, QC, CP, VM, ANS)
C     Equation (A-9).
      IMPLICIT NONE
      INTEGER N, I, J, L, M
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8), ANS
      DOUBLE PRECISION A, B, P, Q, C, T
      DOUBLE PRECISION LI, LJ, LL, QI, QJ, QL
      DOUBLE PRECISION CIJ, CIL, CJL
      DOUBLE PRECISION MI2, MI3, MI4, MJ2, MJ3, MJ4, ML2, ML3, ML4

      ANS = 0.0D0

C     Single-variable terms.
      DO 10 I = 1, N
         ANS = ANS + LC(I)**4*VM(I,4)
         ANS = ANS + QC(I)**4*VM(I,8)
         ANS = ANS + 4.0D0*LC(I)**3*QC(I)*VM(I,5)
         ANS = ANS + 4.0D0*LC(I)*QC(I)**3*VM(I,7)
         ANS = ANS + 6.0D0*LC(I)**2*QC(I)**2*VM(I,6)
   10 CONTINUE

C     Pair terms, i < j.
      DO 30 I = 1, N-1
         DO 20 J = I+1, N
            A = LC(I)
            B = LC(J)
            P = QC(I)
            Q = QC(J)
            C = CP(I,J)

            ANS = ANS + 6.0D0*A**2*B**2*VM(I,2)*VM(J,2)
            ANS = ANS + 6.0D0*P**2*Q**2*VM(I,4)*VM(J,4)
            ANS = ANS + C**4*VM(I,4)*VM(J,4)

            T = A**2*B*VM(I,3)*VM(J,2) + A*B**2*VM(I,2)*VM(J,3)
            ANS = ANS + 12.0D0*C*T

            T = P*VM(I,5)*VM(J,3) + Q*VM(I,3)*VM(J,5)
            ANS = ANS + 12.0D0*C*P*Q*T

            T = A**2*VM(I,4)*VM(J,2) + B**2*VM(I,2)*VM(J,4)
            T = T + 2.0D0*A*B*VM(I,3)*VM(J,3)
            ANS = ANS + 12.0D0*P*Q*T
            ANS = ANS + 6.0D0*C**2*T

            T = P**2*VM(I,6)*VM(J,2) + Q**2*VM(I,2)*VM(J,6)
            T = T + 2.0D0*P*Q*VM(I,4)*VM(J,4)
            ANS = ANS + 6.0D0*C**2*T

            T = B*P*(B*VM(I,3)*VM(J,3) + 2.0D0*A*VM(I,4)*VM(J,2))
            T = T + A*Q*(A*VM(I,3)*VM(J,3) + 2.0D0*B*VM(I,2)*VM(J,4))
            ANS = ANS + 12.0D0*C*T

            T = A*Q*(Q*VM(I,2)*VM(J,5) + 2.0D0*P*VM(I,4)*VM(J,3))
            T = T + B*P*(P*VM(I,5)*VM(J,2) + 2.0D0*Q*VM(I,3)*VM(J,4))
            ANS = ANS + 12.0D0*C*T

            T = P*(A*VM(I,5)*VM(J,2) + B*VM(I,4)*VM(J,3))
            T = T + Q*(A*VM(I,3)*VM(J,4) + B*VM(I,2)*VM(J,5))
            ANS = ANS + 12.0D0*C**2*T
   20    CONTINUE
   30 CONTINUE

C     Cross-diagonal pair terms, j /= i.
      DO 50 I = 1, N
         DO 40 J = 1, N
            IF (J .NE. I) THEN
               A = LC(I)
               B = LC(J)
               P = QC(I)
               Q = QC(J)
               C = CP(I,J)

               ANS = ANS + 4.0D0*P**3*Q*VM(I,6)*VM(J,2)
               ANS = ANS + 4.0D0*P*B**3*VM(I,2)*VM(J,3)
               ANS = ANS + 12.0D0*A*P*B**2*VM(I,3)*VM(J,2)
               ANS = ANS + 12.0D0*A*P**2*Q*VM(I,5)*VM(J,2)
               ANS = ANS + 12.0D0*A*P*Q**2*VM(I,3)*VM(J,4)
               ANS = ANS + 4.0D0*A*C**3*VM(I,4)*VM(J,3)
               ANS = ANS + 4.0D0*P*C**3*VM(I,5)*VM(J,3)
               ANS = ANS + 6.0D0*P**2*B**2*VM(I,4)*VM(J,2)
            END IF
   40    CONTINUE
   50 CONTINUE

C     Triplet terms, i < j < l.
      DO 80 I = 1, N-2
         DO 70 J = I+1, N-1
            DO 60 L = J+1, N
               LI = LC(I)
               LJ = LC(J)
               LL = LC(L)
               QI = QC(I)
               QJ = QC(J)
               QL = QC(L)
               CIJ = CP(I,J)
               CIL = CP(I,L)
               CJL = CP(J,L)
               MI2 = VM(I,2)
               MI3 = VM(I,3)
               MI4 = VM(I,4)
               MJ2 = VM(J,2)
               MJ3 = VM(J,3)
               MJ4 = VM(J,4)
               ML2 = VM(L,2)
               ML3 = VM(L,3)
               ML4 = VM(L,4)

               T = 12.0D0*QI**2*QJ*QL + 6.0D0*CIJ**2*CIL**2
               T = T + 12.0D0*QI*(QL*CIJ**2 + QJ*CIL**2)
               T = T + 6.0D0*QI**2*CJL**2
               ANS = ANS + T*MI4*MJ2*ML2

               T = 12.0D0*QI*QJ**2*QL + 6.0D0*CIJ**2*CJL**2
               T = T + 12.0D0*QJ*(QL*CIJ**2 + QI*CJL**2)
               T = T + 6.0D0*QJ**2*CIL**2
               ANS = ANS + T*MI2*MJ4*ML2

               T = 12.0D0*QI*QJ*QL**2 + 6.0D0*CIL**2*CJL**2
               T = T + 12.0D0*QL*(QI*CJL**2 + QJ*CIL**2)
               T = T + 6.0D0*QL**2*CIJ**2
               ANS = ANS + T*MI2*MJ2*ML4

               T = 12.0D0*CIJ**2*CIL*CJL + 24.0D0*QI*QJ*QL*CIJ
               T = T + 4.0D0*QL*CIJ**3 + 24.0D0*QI*QJ*CIL*CJL
               ANS = ANS + T*MI3*MJ3*ML2

               T = 12.0D0*CIJ*CIL**2*CJL + 24.0D0*QI*QJ*QL*CIL
               T = T + 4.0D0*QJ*CIL**3 + 24.0D0*QI*QL*CIJ*CJL
               ANS = ANS + T*MI3*MJ2*ML3

               T = 12.0D0*CIJ*CIL*CJL**2 + 24.0D0*QI*QJ*QL*CJL
               T = T + 4.0D0*QI*CJL**3 + 24.0D0*QJ*QL*CIJ*CIL
               ANS = ANS + T*MI2*MJ3*ML3

               T = LI*MI3*MJ2*ML2 + LJ*MI2*MJ3*ML2 + LL*MI2*MJ2*ML3
               ANS = ANS + 24.0D0*(QI*QJ*QL + CIJ*CIL*CJL)*T

               T = LI*CJL**2*MI2*(CIJ*MJ3*ML2 + CIL*MJ2*ML3)
               T = T + LJ*CIL**2*MJ2*(CIJ*MI3*ML2 + CJL*MI2*ML3)
               T = T + LL*CIJ**2*ML2*(CIL*MI3*MJ2 + CJL*MI2*MJ3)
               ANS = ANS + 12.0D0*T

               T = QI*CJL**2*MI3*(CIJ*MJ3*ML2 + CIL*MJ2*ML3)
               T = T + QJ*CIL**2*MJ3*(CIJ*MI3*ML2 + CJL*MI2*ML3)
               T = T + QL*CIJ**2*ML3*(CIL*MI3*MJ2 + CJL*MI2*MJ3)
               ANS = ANS + 12.0D0*T

               T = QI*MI4*MJ2*ML2 + QJ*MI2*MJ4*ML2 + QL*MI2*MJ2*ML4
               ANS = ANS + 24.0D0*CIJ*CIL*CJL*T

               T = 12.0D0*(QI*QJ*LL**2 + QI*QL*LJ**2 + QJ*QL*LI**2)
               T = T + 6.0D0*(LI**2*CJL**2 + LJ**2*CIL**2
     &                        + LL**2*CIJ**2)
               T = T + 24.0D0*(CIJ*CIL*LJ*LL + CIJ*CJL*LI*LL
     &                        + CIL*CJL*LI*LJ)
               T = T + 24.0D0*(LI*LJ*QL*CIJ + LI*LL*QJ*CIL
     &                        + LJ*LL*QI*CJL)
               ANS = ANS + T*MI2*MJ2*ML2

               T = 24.0D0*LJ*CIJ*QI*QL + 24.0D0*LL*CIL*QI*QJ
               T = T + 12.0D0*LI*CJL**2*QI + 24.0D0*LJ*CIL*CJL*QI
               T = T + 24.0D0*LL*CIJ*CJL*QI + 12.0D0*LI*CIL**2*QJ
               T = T + 12.0D0*LI*CIJ**2*QL
               ANS = ANS + T*MI3*MJ2*ML2

               T = 24.0D0*LI*CIJ*QJ*QL + 24.0D0*LL*CJL*QI*QJ
               T = T + 12.0D0*LJ*CIL**2*QJ + 24.0D0*LI*CIL*CJL*QJ
               T = T + 24.0D0*LL*CIJ*CIL*QJ + 12.0D0*LJ*CJL**2*QI
               T = T + 12.0D0*LJ*CIJ**2*QL
               ANS = ANS + T*MI2*MJ3*ML2

               T = 24.0D0*LI*CIL*QJ*QL + 24.0D0*LJ*CJL*QI*QL
               T = T + 12.0D0*LL*CIJ**2*QL + 24.0D0*LI*CIJ*CJL*QL
               T = T + 24.0D0*LJ*CIJ*CIL*QL + 12.0D0*LL*CJL**2*QI
               T = T + 12.0D0*LL*CIL**2*QJ
               ANS = ANS + T*MI2*MJ2*ML3
   60       CONTINUE
   70    CONTINUE
   80 CONTINUE

C     Quadruplet terms, i < j < l < m.
      DO 120 I = 1, N-3
         DO 110 J = I+1, N-2
            DO 100 L = J+1, N-1
               DO 90 M = L+1, N
                  T = 24.0D0*(QC(I)*QC(J)*QC(L)*QC(M)
     &                + CP(I,J)*CP(I,L)*CP(J,M)*CP(L,M)
     &                + CP(I,J)*CP(I,M)*CP(J,L)*CP(L,M)
     &                + CP(I,L)*CP(I,M)*CP(J,L)*CP(J,M)
     &                + QC(I)*CP(J,L)*CP(J,M)*CP(L,M)
     &                + QC(J)*CP(I,L)*CP(I,M)*CP(L,M)
     &                + QC(L)*CP(I,J)*CP(I,M)*CP(J,M)
     &                + QC(M)*CP(I,J)*CP(I,L)*CP(J,L))
                  T = T + 12.0D0*(QC(I)*QC(J)*CP(L,M)**2
     &                + QC(I)*QC(L)*CP(J,M)**2
     &                + QC(I)*QC(M)*CP(J,L)**2
     &                + QC(J)*QC(L)*CP(I,M)**2
     &                + QC(J)*QC(M)*CP(I,L)**2
     &                + QC(L)*QC(M)*CP(I,J)**2)
                  T = T + 6.0D0*(CP(I,J)**2*CP(L,M)**2
     &                + CP(I,L)**2*CP(J,M)**2
     &                + CP(I,M)**2*CP(J,L)**2)
                  ANS = ANS + T*VM(I,2)*VM(J,2)*VM(L,2)*VM(M,2)
   90          CONTINUE
  100       CONTINUE
  110    CONTINUE
  120 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE CENMOM(VY, K, ANS)
C     Kth central moment from the raw moments VY(0:4), 0 <= K <= 4.
      IMPLICIT NONE
      INTEGER K
      DOUBLE PRECISION VY(0:4), ANS

      ANS = 0.0D0
      IF (K .EQ. 0) THEN
         ANS = 1.0D0
      ELSE IF (K .EQ. 1) THEN
         ANS = 0.0D0
      ELSE IF (K .EQ. 2) THEN
         ANS = VY(2) - VY(1)**2
      ELSE IF (K .EQ. 3) THEN
         ANS = VY(3) - 3.0D0*VY(2)*VY(1) + 2.0D0*VY(1)**3
      ELSE IF (K .EQ. 4) THEN
         ANS = VY(4) - 4.0D0*VY(3)*VY(1) + 6.0D0*VY(2)*VY(1)**2
     &         - 3.0D0*VY(1)**4
      END IF

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE MOMENT(N, LC, QC, CP, VM, VY, VZ)
C     All raw moments VY(0:4) and central moments VZ(0:4) of y.
      IMPLICIT NONE
      INTEGER N, K
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8)
      DOUBLE PRECISION VY(0:4), VZ(0:4)

      DO 10 K = 0, 4
         CALL RAWMOM(N, LC, QC, CP, VM, K, VY(K))
   10 CONTINUE

      DO 20 K = 0, 4
         CALL CENMOM(VY, K, VZ(K))
   20 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE VARCMP(N, LC, QC, CP, VM, VZ2, VCLC, VCQC, VCCP,
     &                  WLC, WQC, WCP)
C     Output-variance contribution of every polynomial term, obtained by
C     zeroing one coefficient at a time and differencing the variance.
C     WLC, WQC and WCP are scratch copies supplied by the caller, so this
C     stays within FORTRAN 77 (no automatic arrays) and imposes no limit
C     on the number of input variables.
      IMPLICIT NONE
      INTEGER N, I, J
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8), VZ2
      DOUBLE PRECISION VCLC(N), VCQC(N), VCCP(N,N)
      DOUBLE PRECISION WLC(N), WQC(N), WCP(N,N)
      DOUBLE PRECISION R1, R2, SAVE1, SAVE2

      DO 10 I = 1, N
         WLC(I) = LC(I)
         WQC(I) = QC(I)
   10 CONTINUE
C     VCCP is only written on the i < j triangle below, so zero all of it
C     first: it is an intent(out) argument and would otherwise hand back
C     uninitialised memory off the diagonal.
      DO 30 J = 1, N
         DO 20 I = 1, N
            WCP(I,J) = CP(I,J)
            VCCP(I,J) = 0.0D0
   20    CONTINUE
   30 CONTINUE

      DO 40 I = 1, N
         SAVE1 = WLC(I)
         WLC(I) = 0.0D0
         CALL RAWM1(N, WQC, VM, R1)
         CALL RAWM2(N, WLC, WQC, WCP, VM, R2)
         VCLC(I) = VZ2 - (R2 - R1**2)
         WLC(I) = SAVE1
   40 CONTINUE

      DO 50 I = 1, N
         SAVE1 = WQC(I)
         WQC(I) = 0.0D0
         CALL RAWM1(N, WQC, VM, R1)
         CALL RAWM2(N, WLC, WQC, WCP, VM, R2)
         VCQC(I) = VZ2 - (R2 - R1**2)
         WQC(I) = SAVE1
   50 CONTINUE

      DO 70 I = 1, N-1
         DO 60 J = I+1, N
            SAVE1 = WCP(I,J)
            SAVE2 = WCP(J,I)
            WCP(I,J) = 0.0D0
            WCP(J,I) = 0.0D0
            CALL RAWM1(N, WQC, VM, R1)
            CALL RAWM2(N, WLC, WQC, WCP, VM, R2)
            VCCP(I,J) = VZ2 - (R2 - R1**2)
            WCP(I,J) = SAVE1
            WCP(J,I) = SAVE2
   60    CONTINUE
   70 CONTINUE

      RETURN
      END

C-----------------------------------------------------------------------

      SUBROUTINE SOERPM(N, LC, QC, CP, VM, VY, VZ, VCLC, VCQC, VCCP,
     &                  WLC, WQC, WCP)
C     Single entry point: raw moments, central moments and the full set
C     of variance contributions, so the caller crosses the language
C     boundary once per uncertain result rather than O(N**2) times.
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION LC(N), QC(N), CP(N,N), VM(N,0:8)
      DOUBLE PRECISION VY(0:4), VZ(0:4)
      DOUBLE PRECISION VCLC(N), VCQC(N), VCCP(N,N)
      DOUBLE PRECISION WLC(N), WQC(N), WCP(N,N)

      CALL MOMENT(N, LC, QC, CP, VM, VY, VZ)
      CALL VARCMP(N, LC, QC, CP, VM, VZ(2), VCLC, VCQC, VCCP,
     &            WLC, WQC, WCP)

      RETURN
      END

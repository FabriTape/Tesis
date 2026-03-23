#!/bin/bash

#OUTDIR="./sims_SI_erosion_bed_100um_ff_0.15_vels_50_move_1_NP_1_cores_32"
OUTDIR="./sims_SI_erosion_bed_100um_ff_0.15-0.35_vels_all_move_1_NP_1"
INPUT="collision_erosion_bed_np1.input"

NDUMPS=25000
NTHERMO=5000

np=1

#FILLS=(0.15 0.20 0.25 0.30 0.35 0.40)
FILLS=(0.15 0.35)
#FILLS=(0.15 0.35)
#VELS=(-50)
VELS=(-5 -10 -25 -50)
#VELS=(-200)
#TOTALSTEPS=(1000000 500000 250000 200000 150000 100000)
TOTALSTEPS=( 1000000 1000000 1000000 1000000 )
#TOTALSTEPS=(20000 20000 20000 20000 )
#TOTALSTEPS=(25000000) #For 10 m/s
#TOTALSTEPS=(4000000) #For 200 m/s


#PARA NP1 no hay rotaciones
xrotate=0
yrotate=0
zrotate=0
ANGLES=( 0 )

R=1.52    #RADIO DEL GRANO
MOVE=( 0 )
#MOVE=( -1 0 1 ) #Multiplicador del radio para desplazar centro 
#MOVE=( -4 -3 -2 -1 0 1 2 3 4 ) #Multiplicador del radio para desplazar centro 


XMOVE=()   #NO BORRAR
YMOVE=()

len=${#MOVE[@]}
for (( x=0; x<$len; x++ )); do 	
	xmove=${MOVE[$x]}
	for (( y=0; y<$len; y++ )); do
		ymove=${MOVE[$y]}
		
		xfinal=`echo $xmove*$R | bc`
		yfinal=`echo $ymove*$R | bc`
		
		# No entrar si la columna x es la primera y la ultima, para eliminar 18 puntos y tener un rect de 7x9 puntos
		#if [ $x -gt 0 ] && [ $x -lt $(( len-1 )) ]; then 
			XMOVE+=( $xfinal )
			YMOVE+=( $yfinal )
		#fi
	done	
done

echo "Total amount of simulations: ${#XMOVE[@]}"
#exit 

#4r
#xmove=( 0 3.04 0    -3.04  0    6.5347 -6.5347  6.5347 -6.5347 )
#ymove=( 0 0    3.04  0    -3.04 6.5347  6.5347 -6.5347 -6.5347 )

#xmove=( 0 )
#ymove=( 0 )

#r
#xmove=( 0 0.76 0    -0.76  0    0.4084 -0.4084  0.4084 -0.4084 )
#ymove=( 0 0    0.76  0    -0.76 0.4084  0.4084 -0.4084 -0.4084 )

#r/2
#xmove=( 0 0.36 0    -0.36  0    0.0916 -0.0916  0.0916 -0.0916 )
#ymove=( 0 0    0.36  0    -0.36 0.0916  0.0916 -0.0916 -0.0916 )

#xmove=( 0 1 0 -1 -1 2 -2 )
#ymove=( 0 1 1  1 -1 1  1 )
ZMOVE=0



#N=1 # para GPU
N=32 # para CPU desde 1 a N nucleos disponibles en la maquina donde se corra

EXE="lmp_g++_openmpi-4.0.2_stableSep2021 "
#GPU="-cuda on -sf cuda"
#GPU="-cuda off"  # para correr en CPU descomentar esta linea y comentar la de arriba
MPIRUN="mpirun --allow-run-as-root -np $N ../../$EXE $GPU"

COUNTER=0

if [ ! -d $OUTDIR ]; then
	mkdir $OUTDIR
fi

cd $OUTDIR

cp ../$0 ./

for fill in "${FILLS[@]}"
do
	DATA="box_fill_${fill}_SI.data"
	cp ../datas/$DATA ./ 

	for vel in "${VELS[@]}"
	do

		for angle in "${ANGLES[@]}"
		do
		
			len=${#XMOVE[@]}
			for (( move=0; move<$len; move++ )); do 	
#			for move in $(seq 0 1 ${#xmove[@]}); do 
				xm=${XMOVE[$move]}
				ym=${YMOVE[$move]}
				zm=$ZMOVE

				simdir="erosion_bed_SI_ff_${fill}_np_${np}_vel_${vel}_angle_${angle}_move_${xm}_${ym}_${zm}"
				echo $simdir
				if [ ! -d $simdir ]; then
					mkdir $simdir
					cd $simdir

					cp ../../$INPUT .
			
					nsteps=${TOTALSTEPS[$COUNTER]}
					
					echo "$DATA xrotate $xrotate yrotate $yrotate zrotate $zrotate angle $angle"
					echo "xmove $xm ymove $ym zmove $zm"
					echo "velocity $vel nsteps $nsteps ndumps $NDUMPS nthermo $NTHERMO"

					$MPIRUN -in $INPUT -v data ../$DATA \
							   -v xrotate $xrotate \
							   -v yrotate $yrotate \
							   -v zrotate $zrotate \
							   -v angle $angle \
							   -v xmove $xm"e-6" \
							   -v ymove $ym"e-6" \
							   -v zmove $zm"e-6" \
							   -v velocity $vel \
							   -v nsteps $nsteps \
							   -v ndumps $NDUMPS \
							   -v nthermo $NTHERMO > salida.log
					cd ..
				fi
			done
		done
		COUNTER=$((COUNTER + 1))
	
	done
	COUNTER=0
done

#pragma once

#include "marchee.cuh"

#define ACTIV(x) tanh(x)

#define MAX_Y 2048

#define N N_FLTR

//	Bloque -> Pensee -> 3 sorties
//	Bloque :
//		c0 -> c1 ... -> 1

#define COUCHES (mdl->C_BLOQUE + mdl->C_PENSEE)

typedef struct {
	//	Bloque
	uint  BLOQUES;	
	uint C_BLOQUE;
	uint * BLOQUE; uint * BLOQUE__d;

	ema_int * arr_ema_int;
	ema_int * arr_ema_int__d;

	//	Pensee
	uint C_PENSEE;
	uint * PENSEE; uint * PENSEE__d;

	//	Totaux
	uint FILTRES, POIDS, VARS, LOCDS;
	uint * DEPART_POIDS, * DEPART_VARS, * DEPART_LOCDS;
	uint * DEPART_POIDS__d, * DEPART_VARS__d, * DEPART_LOCDS__d;

	//	[CPU] : BLOQUES + PENSEE
	float * f;
	float * p;
	float * y;
	float * locd;

	//	[GPU] : BLOQUES + PENSEE
	float * f_d;
	float * p_d;

	//	[GPU] : grad_BLOQUES + grad_PENSEE
	float * dp_d;
	float * dy_d;

	//	[CPU & GPU] : Espace Optimisation
	float * dif_f;
	float * dif_f_d;
} Mdl_t;

//	--- Allocation & Gestion Memoire ---
Mdl_t * cree_mdl(
	uint BLOQUES, uint BLOQUE_DIM, uint * BLOQUE,
	ema_int * arr_ema_int,
	uint PENSEE_DIM, uint * PENSEE);
void mdl_liberer(Mdl_t * mdl);
void    prep_mdl(Mdl_t * mdl);
void  reinit_mdl(Mdl_t * mdl);

//	--- Transferts ---
void gpu_vers_cpu(Mdl_t * mdl);

//	--- Plume ---
void   taille_mdl(Mdl_t * mdl);
void    plume_mdl(Mdl_t * mdl);
//
void comportement(Mdl_t * mdl);

//	--- Calc ---
#define CPU 0
float                   f(Mdl_t * mdl, uint t);	//cpu
void       cpu_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);
//	[Méthode 0] Basique copier pareil que CPU 
#define MDT0 1
void cuda_mdt0_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);
//	[Méthode 1] Decomposition en Instructions Individuelles 
#define MDT1 2
void cuda_mdt1_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);
//	[Méthode 2] Methode 2 + Filtres plus Rapides
void cuda_mdt2_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);

//	--- Pred / Gain / Gain_Moyen ---
float pred_mdl(Mdl_t * mdl, uint METHODE, uint t0, uint t1);
/*	0 - CPU
	1 - GPU mdt0
	2 - GPU mdt1
*/
#include "main.cuh"

PAS_OPTIMISER()
void charger() {
	charger_les_prixs();
	calculer_ema_norm_diff();
	charger_vram_nvidia();
}

PAS_OPTIMISER()
void liberer_tout() {
	liberer_cudamalloc();
}

PAS_OPTIMISER()
void charger_tout() {
	printf("Chargement des prixs : ");
	MESURER(charger());

	printf("           prixs = %3.3f Mo\n", ((float)sizeof(float)*PRIXS)                     / 1e6);
	printf("             ema = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*PRIXS)            / 1e6);
	printf("      normalisee = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*N_FLTR*PRIXS)     / 1e6);
	printf("  dif_normalisee = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*(N_FLTR-1)*PRIXS) / 1e6);
};

PAS_OPTIMISER()
void performances() {
#define FB 16 //filtres par bloque
#define BLOQUES 2*2*32
	//ASSERT(FB*BLOQUES <= 1024)
#define C_BLOQUE 5
#define C_PENSEE 7
	uint BLOQUE[C_BLOQUE] = {FB, 16, 8, 4, 1};
	uint PENSEE[C_PENSEE] = {256, 128, 64, 32, 16, 4, 1};

	ema_int EMA_INT[FB*BLOQUES] = {ema_ints[0]};

	Mdl_t * mdl = cree_mdl(
		BLOQUES, C_BLOQUE, BLOQUE,
		EMA_INT,
		C_PENSEE, PENSEE
	);
	taille_mdl(mdl);
	plume_mdl(mdl);

	//	-----------------------------------------------
	INIT_CHRONO(a)
	float pred;
	float perf;
	float gpu_perf;

	//	-----------------------------------------------
	DEPART_CHRONO(a)
#define REP_cpu 1
	FOR(0, i, REP_cpu) pred = pred_mdl(mdl, 0, DEPART, FIN);
	perf = REP_cpu / (VALEUR_CHRONO(a));
	printf(
		"pred=%+f;  temps %i rep ~= %3.3f  [%3.3f cpu_pred_mdl(mdl) par seconde]\n",
		pred, REP_cpu, VALEUR_CHRONO(a), perf
	);

	//	-----------------------------------------------
	FOR(0, METHODE, 3) {
		DEPART_CHRONO(a)
	#define REP_gpu 1
		FOR(0, i, REP_gpu) pred = pred_mdl(mdl, 1+METHODE, DEPART, FIN);
		gpu_perf = REP_gpu / VALEUR_CHRONO(a);
		printf(
			"pred=%+f;  temps %i rep ~= %3.3f  [%3.3f gpu_pred_mdl(mdl) par seconde (~ %3.3fx plus rapide)] [METHODE=%i]\n",
			pred, REP_gpu, VALEUR_CHRONO(a), gpu_perf, gpu_perf/perf,
			METHODE
		);
	}

	//	Fin
	//comportement(mdl);
	//plume_mdl(mdl);
	mdl_liberer(mdl);
};

PAS_OPTIMISER()
void verif_df() {
	
};

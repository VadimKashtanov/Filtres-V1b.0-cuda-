#include "mdl.cuh"

PAS_OPTIMISER()
Mdl_t * cree_mdl(
	uint BLOQUES, uint BLOQUE_DIM, uint * BLOQUE,
	ema_int * arr_ema_int,
	uint PENSEE_DIM, uint * PENSEE)
{
	ASSERT(BLOQUE[BLOQUE_DIM-1] == 1);
	//ASSERT(PENSEE[PENSEE_DIM-1] == 3);
	Mdl_t * mdl = (Mdl_t*)malloc(sizeof(Mdl_t));

	mdl->BLOQUES = BLOQUES;
	mdl->C_BLOQUE = BLOQUE_DIM;
	mdl->BLOQUE = cpyuint(BLOQUE, BLOQUE_DIM);

	mdl->arr_ema_int = (ema_int*)malloc(sizeof(ema_int) * BLOQUE[0] * BLOQUES);
	memcpy(mdl->arr_ema_int, arr_ema_int, sizeof(ema_int) * BLOQUE[0] * BLOQUES);

	mdl->C_PENSEE = PENSEE_DIM;
	mdl->PENSEE = cpyuint(PENSEE, PENSEE_DIM);

	mdl->FILTRES = BLOQUE[0] * BLOQUES * N;
	mdl->POIDS = 0;
	mdl->VARS  = 0;
	mdl->LOCDS = 0;

	mdl->DEPART_POIDS = (uint*)malloc(sizeof(uint) * COUCHES);
	mdl->DEPART_VARS  = (uint*)malloc(sizeof(uint) * COUCHES);
	mdl->DEPART_LOCDS = (uint*)malloc(sizeof(uint) * COUCHES);

	//	Instruction: Bloque
	FOR(0, i, mdl->C_BLOQUE) {
		ASSERT(mdl->BLOQUE[i]*mdl->BLOQUES <= MAX_Y);
		mdl->DEPART_VARS [i] = mdl->VARS ;
		mdl->DEPART_POIDS[i] = mdl->POIDS;
		mdl->DEPART_LOCDS[i] = mdl->LOCDS;
		//
		mdl->VARS  +=                 BLOQUE[i]                * BLOQUES;
		mdl->POIDS += (i == 0 ? 0 : ((BLOQUE[i-1]+1)*BLOQUE[i])) * BLOQUES;
		mdl->LOCDS += (        i == 0 ? 0 : BLOQUE[i]        ) * BLOQUES;
	};

	//	Instruction: Pensee
	FOR(0, i, mdl->C_PENSEE) {
		ASSERT(mdl->PENSEE[i] <= MAX_Y);

		mdl->DEPART_VARS [mdl->C_BLOQUE+i] = mdl->VARS ;
		mdl->DEPART_POIDS[mdl->C_BLOQUE+i] = mdl->POIDS;
		mdl->DEPART_LOCDS[mdl->C_BLOQUE+i] = mdl->LOCDS;
		//
		//printf("vars=%i +%i\n", mdl->VARS, PENSEE[i]);
		mdl->VARS  += PENSEE[i];
		mdl->POIDS += (i == 0 ? ((BLOQUES+1)*PENSEE[0]) : ((PENSEE[i-1]+1)*PENSEE[i]));
		mdl->LOCDS += PENSEE[i];
	};

	//	======= Allocation ========
	mdl->f    = (float*)malloc(sizeof(float) * mdl->FILTRES);
	mdl->p    = (float*)malloc(sizeof(float) * mdl->POIDS);
	mdl->y    = (float*)malloc(sizeof(float) * mdl->VARS);
	mdl->locd = (float*)malloc(sizeof(float) * mdl->LOCDS);

	CONTROLE_CUDA(cudaMalloc((void**)&mdl->f_d,    sizeof(float) * mdl->FILTRES));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->p_d,    sizeof(float) * mdl->POIDS));
	//CONTROLE_CUDA(cudaMalloc((void**)&mdl->y_d,    sizeof(float) * mdl->VARS));
	//CONTROLE_CUDA(cudaMalloc((void**)&mdl->locd_d, sizeof(float) * mdl->LOCDS));

	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dp_d, sizeof(float) * mdl->POIDS);)
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dy_d, sizeof(float) * mdl->VARS));

	mdl->dif_f = (float*)malloc(sizeof(float) * BLOQUE[0] * BLOQUES * (N-1));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dif_f_d, sizeof(float) * BLOQUE[0] * BLOQUES * (N-1)));

	FOR(0, i, mdl->FILTRES) mdl->f[i] = rnd();
	FOR(0, i, mdl->POIDS) mdl->p[i] = 2*rnd()-1;
	FOR(0, i, mdl->FILTRES / N) normer(mdl->f + i*N, N);

	//	Qlq uint pour cuda
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->DEPART_POIDS__d, sizeof(float)*(mdl->C_BLOQUE+mdl->C_PENSEE)));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->PENSEE__d, sizeof(float)*mdl->C_PENSEE));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->BLOQUE__d, sizeof(float)*mdl->C_BLOQUE));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->arr_ema_int__d, sizeof(ema_int)*mdl->BLOQUES*mdl->BLOQUE[0]));
	//
	CONTROLE_CUDA(cudaMemcpy(mdl->DEPART_POIDS__d, mdl->DEPART_POIDS, sizeof(float)*(mdl->C_BLOQUE+mdl->C_PENSEE), cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->PENSEE__d, 	  mdl->PENSEE,       sizeof(float)*mdl->C_PENSEE, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->BLOQUE__d,       mdl->BLOQUE,       sizeof(float)*mdl->C_BLOQUE, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->arr_ema_int__d,  mdl->arr_ema_int,  sizeof(ema_int)*mdl->BLOQUES*mdl->BLOQUE[0], cudaMemcpyHostToDevice));
	//

	prep_mdl(mdl);

	return mdl;
};

void mdl_liberer(Mdl_t * mdl) {
	free(mdl->BLOQUE);
	free(mdl->PENSEE);
	free(mdl->DEPART_POIDS);
	free(mdl->DEPART_VARS);
	free(mdl->DEPART_LOCDS);
	//
	free(mdl->arr_ema_int);
	//
	free(mdl->dif_f);
	//
	free(mdl->f);
	free(mdl->p);
	free(mdl->y);
	free(mdl->locd);
	//
	CONTROLE_CUDA(cudaFree(mdl->f_d));
	CONTROLE_CUDA(cudaFree(mdl->p_d));
	//CONTROLE_CUDA(cudaFree(mdl->y_d));
	//CONTROLE_CUDA(cudaFree(mdl->locd_d));
	CONTROLE_CUDA(cudaFree(mdl->dp_d));
	CONTROLE_CUDA(cudaFree(mdl->dy_d));
	CONTROLE_CUDA(cudaFree(mdl->dif_f_d));
	//
	CONTROLE_CUDA(cudaFree(mdl->DEPART_POIDS__d));
	CONTROLE_CUDA(cudaFree(mdl->PENSEE__d));
	CONTROLE_CUDA(cudaFree(mdl->BLOQUE__d));
	CONTROLE_CUDA(cudaFree(mdl->arr_ema_int__d));
};

void prep_mdl(Mdl_t * mdl) {
	memset(mdl->y, 0, sizeof(float) * mdl->VARS);
	memset(mdl->locd, 0, sizeof(float) * mdl->LOCDS);

	CONTROLE_CUDA(cudaMemcpy(mdl->p_d, mdl->p, sizeof(float)*mdl->POIDS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->f_d, mdl->f, sizeof(float)*mdl->FILTRES, cudaMemcpyHostToDevice));

	//CONTROLE_CUDA(cudaMemset(mdl->y_d,     0, sizeof(float) * mdl->VARS));
	//CONTROLE_CUDA(cudaMemset(mdl->locd_d,  0, sizeof(float) * mdl->LOCDS));
	CONTROLE_CUDA(cudaMemset(mdl->dp_d,    0, sizeof(float) * mdl->POIDS));
	CONTROLE_CUDA(cudaMemset(mdl->dy_d,    0, sizeof(float) * mdl->VARS));
	
	uint i;
//#pragma omp parallel for private(i)
	for (i=0; i < mdl->BLOQUE[0] * mdl->BLOQUES; i++) {
		FOR(0, j, N-1) {
			mdl->dif_f[i*(N-1) + j] = mdl->f[i*N+j+1]-mdl->f[i*N+j];
		}
	}
	cudaMemcpy(mdl->dif_f_d, mdl->dif_f, sizeof(float)*mdl->BLOQUE[0] * mdl->BLOQUES * (N-1), cudaMemcpyHostToDevice);
};

void gpu_vers_cpu(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaMemcpy(mdl->p, mdl->p_d, sizeof(float)*mdl->POIDS, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->f, mdl->f_d, sizeof(float)*mdl->FILTRES, cudaMemcpyDeviceToHost));
};

void reinit_mdl(Mdl_t * mdl) {
	prep_mdl(mdl);
};
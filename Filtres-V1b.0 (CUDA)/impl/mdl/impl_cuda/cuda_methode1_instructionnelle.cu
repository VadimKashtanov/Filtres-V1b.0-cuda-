#include "mdl.cuh"

static __global__ void cart_graphique_kerd__FILTRES(
	float * y_d, uint C_MAX,
	ema_int * arr_ema_int__d,
	float * normalisee__d, float * dif_normalisee__d,
	float * f_d, float * dif_f_d,
	uint FS, uint TS,
	uint t0)
{
	// <<< (filtre, t) >>>
	uint Fx = threadIdx.x + blockIdx.x * blockDim.x;
	uint Ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (Ty < TS && Fx < FS) {
		uint t = t0 + Ty;
		//
		float s = 0, d = 0;
		uint ligne = arr_ema_int__d[Fx].ligne;
		FOR(0, i, N-1) {
			s += sqrtf((1.f + fabs(
				    normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + i] - /*const_mem[Fx*N+i]*/f_d[Fx*N+i]
			)));
			d += powf((1.f + fabs(
				dif_normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + i] - /*const_mem[FS*N + Fx*(N-1)+i]*/dif_f_d[Fx*(N-1)+i]
			)), 2);
		};
		s += sqrtf(1.f + fabs(
			normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + (N-1)] - /*const_mem[Fx*N + (N-1)]*/f_d[Fx*N + (N-1)]
		));
		s = s/8-1;
		d = d/7-1;
		//
		y_d[Ty*C_MAX + Fx] = 2*expf(-s*s -d*d)-1;
	}
};

static __global__ void cart_graphique_kerd__perceptron_BLOQUE(
	uint * DEPART_POIDS__d, uint c,
	uint BLOQUES, uint X, uint Y,
	float * x_d, float * y_d, uint C_MAX,
	float * p_d,
	uint TS)
{
	//	<<<(Bloque, t, y)>>>
	uint Bx = threadIdx.x + blockIdx.x * blockDim.x;
	uint Ty = threadIdx.y + blockIdx.y * blockDim.y;
	uint Yz = threadIdx.z + blockIdx.z * blockDim.z;

	if (((Bx < BLOQUES) && (Ty < TS)) && (Yz < Y)) {
		float s = p_d[DEPART_POIDS__d[c] + Bx*Y*(X+1) + Yz*(X+1) + (X-1)+1];
		FOR(0, i, X)
			s += x_d[Ty*C_MAX + Bx*X + i] * p_d[DEPART_POIDS__d[c] + Bx*Y*(X+1) + Yz*(X+1) + i];
		y_d[Ty*C_MAX + Bx*Y + Yz] = tanh(s);
	}
}

static __global__ void cart_graphique_kerd__perceptron_PENSEE(
	uint X, uint Y,
	float * x_d, float * y_d, uint C_MAX,
	float * p_d,
	uint TS)
{
	// <<< (y, t) >>>
	uint Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint Ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (Ty < TS && Yx < Y) {
		float s = p_d[Yx*(X+1) + (X-1)+1];
		FOR(0, i, X) s += x_d[Ty*C_MAX + i] * p_d[Yx*(X+1) + i];
		y_d[Ty*C_MAX + Yx] = tanh(s);
		//printf("%f ", tanh(s));
	}
}

static __global__ void enregistrer_les_resultats(
	float * res_d, float * y,
	uint C_MAX, uint TS)
{
	uint Tx = threadIdx.x + blockIdx.x * blockDim.x;

	if (Tx < TS) {
		res_d[Tx] = y[Tx*C_MAX + 0];
	};
};

void cuda_mdt1_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	float * res_d;
	CONTROLE_CUDA(cudaMalloc((void**)&res_d, sizeof(float)*(t1-t0)));
	CONTROLE_CUDA(cudaMemset(res_d, 0, sizeof(float)*(t1-t0)));
	
	//
	uint C_MAX = mdl->BLOQUE[0] * mdl->BLOQUES;
	FOR(1, i, mdl->C_BLOQUE) if (mdl->BLOQUE[i]*mdl->BLOQUES > C_MAX) C_MAX = mdl->BLOQUE[i]*mdl->BLOQUES;
	FOR(0, i, mdl->C_PENSEE) if (mdl->PENSEE[i] > C_MAX) C_MAX = mdl->PENSEE[i];

	float * r0__d;
	float * r1__d;
	CONTROLE_CUDA(cudaMalloc((void**)&r0__d, sizeof(float) * PRIXS * C_MAX));
	CONTROLE_CUDA(cudaMalloc((void**)&r1__d, sizeof(float) * PRIXS * C_MAX));

	//CONTROLE_CUDA(cudaMemset(r0__d, 0, sizeof(float) * PRIXS * C_MAX));
	//CONTROLE_CUDA(cudaMemset(r1__d, 0, sizeof(float) * PRIXS * C_MAX));

	//	--- Filtres ---
	uint FS = mdl->BLOQUES * mdl->BLOQUE[0];
	uint TS = t1 - t0;
	cart_graphique_kerd__FILTRES<<<dim3(KER_DIV(FS, 16), KER_DIV(TS, 32)),dim3(16,32)>>>(
		r0__d, C_MAX,
		mdl->arr_ema_int__d,
		normalisee__d, dif_normalisee__d,
		mdl->f_d, mdl->dif_f_d,
		FS, TS,
		t0
	);
	ATTENDRE_KER_CUDA();

	//	--- Perceptrons des Bloques ---
	FOR(1, c, mdl->C_BLOQUE) {
		cart_graphique_kerd__perceptron_BLOQUE<<<dim3(KER_DIV(mdl->BLOQUES, 8), KER_DIV(TS, 8), KER_DIV(mdl->BLOQUE[c], 8)), dim3(8,8,8)>>>(
			mdl->DEPART_POIDS__d, c,
			mdl->BLOQUES, mdl->BLOQUE[c-1], mdl->BLOQUE[c],
			(c%2==0 ? r1__d : r0__d), (c%2==0 ? r0__d : r1__d), C_MAX,
			mdl->p_d,// + mdl->DEPART_POIDS[c],
			TS
		);
		ATTENDRE_KER_CUDA();
		//printf("\n");
	}

	//	--- Perceptron de la Pensee ---
	FOR(0, c, mdl->C_PENSEE) {
		cart_graphique_kerd__perceptron_PENSEE<<<dim3(KER_DIV(mdl->PENSEE[c], 16), KER_DIV(TS, 64)),dim3(16,64)>>>(
			(c==0 ? mdl->BLOQUES : mdl->PENSEE[c-1]), mdl->PENSEE[c],
			((mdl->C_BLOQUE+c)%2==0 ? r1__d : r0__d), ((mdl->C_BLOQUE+c)%2==0 ? r0__d : r1__d), C_MAX,
			mdl->p_d + mdl->DEPART_POIDS[mdl->C_BLOQUE + c],
			TS
		);
		ATTENDRE_KER_CUDA();
	}

	//	--- Resultat ---
	enregistrer_les_resultats<<<dim3(KER_DIV(TS,1024)), dim3(1024)>>>(
		res_d, ((mdl->C_BLOQUE+mdl->C_PENSEE)%2==0 ? r1__d : r0__d),
		C_MAX, TS);
	ATTENDRE_KER_CUDA();

	//	Sortie
	CONTROLE_CUDA(cudaMemcpy(
		res,
		res_d,
		sizeof(float)*(t1-t0),
		cudaMemcpyDeviceToHost
	));
	CONTROLE_CUDA(cudaFree(res_d));
	CONTROLE_CUDA(cudaFree(r0__d));
	CONTROLE_CUDA(cudaFree(r1__d));
};
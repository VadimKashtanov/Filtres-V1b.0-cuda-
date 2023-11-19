#include "mdl.cuh"

/*	Filtres : Memoire constante
	Poids   : Memoire constante
*/

__device__ static float perceptron(float * x, float * p, uint _N) {
	float s = p[_N-1+1];
	FOR(0, i, _N) {
		s += x[i]*p[i];
	}
	return ACTIV(s);
};

__global__ void kerd_mdl(
	uint  BLOQUES,	
	uint C_BLOQUE,
	uint * BLOQUE__d,
	ema_int * arr_ema_int__d,
	uint C_PENSEE,
	uint * PENSEE__d,
	uint * DEPART_POIDS__d, uint * DEPART_VARS__d, uint * DEPART_LOCDS__d,
	float * f_d, float * p_d,
	float * dif_f_d,
	uint t0, uint t1, float * res_d,
	float * normalisee__d, float * dif_normalisee__d)
{
	uint t = t0 + (threadIdx.x + blockIdx.x * blockDim.x);
	//
	if (t < t1) {
		float r0[MAX_Y];
		float r1[MAX_Y];

		//	FILTRES
		float s,d;
		uint ligne;
		FOR(0, f, BLOQUES*BLOQUE__d[0]) {
			ligne = arr_ema_int__d[f].ligne;
			s = 0; d = 0;
			FOR(0, i, N-1) {
				s += sqrtf(1 + fabs(
					    normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + i] - f_d[f*N+i]
				));
				d += powf((1 + fabs(
					dif_normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + i] - dif_f_d[f*(N-1)+i]
				)), 2);
			};
			s += sqrtf(1 + fabs(
				normalisee__d[ligne*PRIXS*N_FLTR + t*N_FLTR + (N-1)] - f_d[f*N + (N-1)]
			));
			s = s/8-1;
			d = d/7-1;
			r0[f] = 2*expf(-s*s -d*d)-1;
		}
		
		//	Triangles des Bloques
		FOR(1, c, C_BLOQUE) {
			FOR(0, b, BLOQUES) {
				FOR(0, y, BLOQUE__d[c]) {
					r1[b*BLOQUE__d[c] + y] = perceptron(
						r0 + b*BLOQUE__d[c-1],
						p_d + DEPART_POIDS__d[c] + (b*BLOQUE__d[c] + y)*(BLOQUE__d[c-1]+1),
						BLOQUE__d[c-1]
					);
				};
			}
			FOR(0, y, BLOQUE__d[c] * BLOQUES) {
				r0[y] = r1[y];
			}
		};

		//	Pensee
		FOR(0, c, C_PENSEE) {
			FOR(0, y, PENSEE__d[c]) {
				uint qt_vars_couche_pred = (c == 0 ? BLOQUES*1 : PENSEE__d[c-1]);
				r1[y] = perceptron(
					r0,
					p_d + DEPART_POIDS__d[C_BLOQUE+c] + y*(qt_vars_couche_pred+1),
					(c == 0 ? BLOQUES : PENSEE__d[c-1])
				);
			};
			FOR(0, y, PENSEE__d[c]) r0[y] = r1[y];
		};
		res_d[t-t0] = r0[0];
	};
};

void cuda_mdt0_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	float * res_d;
	CONTROLE_CUDA(cudaMalloc((void**)&res_d, sizeof(float)*(t1-t0)));
	CONTROLE_CUDA(cudaMemset(res_d, 0, sizeof(float)*(t1-t0)));

	//	--- Mdl_t ---
	kerd_mdl<<<dim3(KER_DIV((t1-t0), 128)), dim3(128)>>>(
		mdl->BLOQUES,	
		mdl->C_BLOQUE,
		mdl->BLOQUE__d,
		mdl->arr_ema_int__d,
		mdl->C_PENSEE,
		mdl->PENSEE__d,
		mdl->DEPART_POIDS__d, mdl->DEPART_VARS__d, mdl->DEPART_LOCDS__d,
		mdl->f_d,
		mdl->p_d,
		mdl->dif_f_d,
		t0, t1, res_d,
		normalisee__d, dif_normalisee__d
	);
	ATTENDRE_KER_CUDA();

	CONTROLE_CUDA(cudaMemcpy(
		res,
		res_d,
		sizeof(float)*(t1-t0),
		cudaMemcpyDeviceToHost
	));

	//
	CONTROLE_CUDA(cudaFree(res_d));
};
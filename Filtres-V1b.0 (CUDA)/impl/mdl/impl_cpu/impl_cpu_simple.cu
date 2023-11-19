#include "mdl.cuh"

static float filtre(float * x, float * dif_x, float * f, float * dif_f) {
	float s = 0, d = 0;
	FOR(0, i, N-1) {
		s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
	};
	s += sqrtf(1 + fabs(x[N-1] - f[N-1]));

	s = s/8-1;
	d = d/7-1;

	return 2*expf(-s*s -d*d)-1;
};

static float perceptron(float * x, float * p, uint _N) {
	float s = p[_N-1+1];
	FOR(0, i, _N) s += x[i]*p[i];
	return ACTIV(s);
};

float f(Mdl_t * mdl, uint t) {
	//	Filtres
	FOR(0, i, mdl->BLOQUE[0] * mdl->BLOQUES) {
		mdl->y[i] = filtre(
			    normalisee + mdl->arr_ema_int[i].ligne*PRIXS*N_FLTR + t*N_FLTR,
			dif_normalisee + mdl->arr_ema_int[i].ligne*PRIXS*N_FLTR + t*N_FLTR,
			mdl->f + i*N,
			mdl->dif_f + i*(N-1)
		);
	}
	
	//	Bloques
	FOR(1, c, mdl->C_BLOQUE) {
		FOR(0, b, mdl->BLOQUES) {
			FOR(0, y, mdl->BLOQUE[c]) {
				mdl->y[mdl->DEPART_VARS[c] + b*mdl->BLOQUE[c] + y] = perceptron(
					mdl->y + mdl->DEPART_VARS[c-1] + b*mdl->BLOQUE[c-1],
					mdl->p +  mdl->DEPART_POIDS[c] + (b*mdl->BLOQUE[c] + y)*(mdl->BLOQUE[c-1]+1),
					mdl->BLOQUE[c-1]
				);
				//printf("%f ", mdl->y[mdl->DEPART_VARS[c] + b*mdl->BLOQUE[c] + y]);
			};
		};
		//printf("\n");
	};

	//	Pensee
	FOR(0, c, mdl->C_PENSEE) {
		FOR(0, y, mdl->PENSEE[c]) {
			uint qt_vars_couche_pred = (c == 0 ? mdl->BLOQUES*1 : mdl->PENSEE[c-1]);
			mdl->y[mdl->DEPART_VARS[mdl->C_BLOQUE+c] + y] = perceptron(
				mdl->y + mdl->DEPART_VARS[mdl->C_BLOQUE+c-1],
				mdl->p + mdl->DEPART_POIDS[mdl->C_BLOQUE+c] + y*(qt_vars_couche_pred+1),
				(c == 0 ? mdl->BLOQUES : mdl->PENSEE[c-1])
			);
		};
	};
	return mdl->y[mdl->VARS-1];
};

void  cpu_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	uint t;
//#pragma omp parallel for private(t)  //(Incoherances et inexactitudes !)
	for (t=t0; t < t1; t++) {
		res[t-t0] = f(mdl, t);
	}
};

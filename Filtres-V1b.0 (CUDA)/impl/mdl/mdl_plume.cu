#include "mdl.cuh"

void taille_mdl(Mdl_t * mdl) {
	//printf("%i %i %i %i\n", mdl->FILTRES, mdl->POIDS, mdl->VARS, mdl->LOCDS);
	printf("  sizeof(Mdl_t) ~= %3.3f Mo\n",
		(float)sizeof(float) * (mdl->FILTRES + mdl->POIDS + mdl->VARS + mdl->LOCDS) / 1e6
	);
};

void plume_mdl(Mdl_t * mdl) {
	printf("Mdl_t %i-bloque %i-pensee (VARS=%i FILTRES=%i POIDS=%i LOCDS=%i)\n",
		mdl->C_BLOQUE, mdl->C_PENSEE,
		mdl->VARS, mdl->FILTRES, mdl->POIDS, mdl->LOCDS);
	printf("%i Bloques (entrees : %i filtres, %i-points) :\n", mdl->BLOQUES, mdl->FILTRES/N, N);
	printf(" 0| filtre [%3.i]*%i DEPART_VARS=%i\n",
		mdl->BLOQUE[0], mdl->BLOQUES,
		mdl->DEPART_VARS[0]
	);
	FOR(1, i, mdl->C_BLOQUE) {
		printf("%2.i| bloque [%3.i]*%i DEPART_VARS=%i DEPART_POIDS=%i DEPART_LOCDS=%i  (poids=%i)\n",
			i, mdl->BLOQUE[i], mdl->BLOQUES,
			mdl->DEPART_VARS[i],
			mdl->DEPART_POIDS[i],
			mdl->DEPART_LOCDS[i],

			mdl->DEPART_POIDS[i+1]-mdl->DEPART_POIDS[i]
		);
	}
	printf("Pensee (entree : bloque[c=%i] y=%i) :\n", mdl->C_BLOQUE-1, 1*mdl->BLOQUES);
	FOR(0, i, mdl->C_PENSEE) {
		printf("%2.i| pensee [%3.i]    DEPART_VARS=%i DEPART_POIDS=%i DEPART_LOCDS=%i  (poids=%i)\n",
			mdl->C_BLOQUE+i,
			mdl->PENSEE[i],
			mdl->DEPART_VARS[mdl->C_BLOQUE+i],
			mdl->DEPART_POIDS[mdl->C_BLOQUE+i],
			mdl->DEPART_LOCDS[mdl->C_BLOQUE+i],

			mdl->DEPART_POIDS[mdl->C_BLOQUE+i+1]-mdl->DEPART_POIDS[mdl->C_BLOQUE+i]
		);
	}
	//float mf=0;
	//FOR(0, i, mdl->FILTRES) mf += mdl->f[i];
	//printf("moyenne filtre = %f\n", mf / (float)mdl->FILTRES);
};
#include "main.cuh"

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	charger_tout();
	performances();
	verif_df();

	//===============

	//gnuplot(prixs, 1000, "prixs");

	//	-- Fin --
	liberer_tout();
};

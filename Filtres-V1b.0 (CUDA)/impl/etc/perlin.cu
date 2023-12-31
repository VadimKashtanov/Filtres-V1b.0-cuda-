#include "etc.cuh"

float noise(int x, int y) {
    int n;

    n = x + y * 57;
    n = (n << 13) ^ n;
    return (1.0 - ( (n * ((n * n * 15731) + 789221) +  1376312589) & 0x7fffffff) / 1073741824.0);
}

float interpolate(float a, float b, float x)
{
  float     pi_mod;
  float     f_unk;

  pi_mod = x * 3.1415927;
  f_unk = (1 - cos(pi_mod)) * 0.5;
  return (a * (1 - f_unk) + b * x);
}

float smooth_noise(int x, int y)
{
  float     corners;
  float     center;
  float     sides;

  corners = (noise(x - 1, y - 1) + noise(x + 1, y - 1) +
         noise(x - 1, x + 1) + noise(x + 1, y + 1)) / 16;
  sides = (noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) +
       noise(x, y + 1)) / 8;
  center = noise(x, y) / 4;
  return (corners + sides + center);
}


float noise_handler(float x, float y)
{
  int       int_val[2];
  float     frac_val[2];
  float     value[4];
  float     res[2];

  int_val[0] = (int)x;
  int_val[1] = (int)y;
  frac_val[0] = x - int_val[0];
  frac_val[1] = y - int_val[1];
  value[0] = smooth_noise(int_val[0], int_val[1]);
  value[1] = smooth_noise(int_val[0] + 1, int_val[1]);
  value[2] = smooth_noise(int_val[0], int_val[1] + 1);
  value[3] = smooth_noise(int_val[0] + 1, int_val[1] + 1);
  res[0] = interpolate(value[0], value[1], frac_val[0]);
  res[1] = interpolate(value[2], value[3], frac_val[0]);
  return (interpolate(res[0], res[1], frac_val[1]));
}

float bruit_de_perlin(float x, float y) {// float gain, int octaves, int hgrid) {
	float gain = 0.5;	//0.6
	int octaves = 3;	//5
	int hgrid = 2;		//2
	//
    int i;
    float total = 0.0f;
    float frequency = 1.0f/(float)hgrid;
    float amplitude = gain;
    float lacunarity = 2.0;

    for (i = 0; i < octaves; ++i)
    {
        total += noise_handler((float)x * frequency, (float)y * frequency) * amplitude;         
        frequency *= lacunarity;
        amplitude *= gain;
    } 

    return (total);
}

void perlin_carteXY(float * carte, uint X, uint Y) {
	uint X_alea=rand()%1000, Y_alea=rand()%1000;
	FOR(0, y, Y) {
		FOR(0, x, X) {
			carte[y*X + x] = bruit_de_perlin(X_alea+x, Y_alea+y);
		}
	}
};
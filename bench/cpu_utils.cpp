static unsigned int g_seed;

float drand48(void) {
	g_seed = (214013 * g_seed + 2531011);
	return (float)((g_seed >> 16) & 0x7FFF) / 32767;
}
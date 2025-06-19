from scipy.optimize import curve_fit
from numba import njit
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import gzip

def linearFunc(x, a, b):
	return a * x + b

def gaussian(x, amp, mu, sigma):
	return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def tripleGaussian(x, amp1, mu1, sig1, amp2, mu2, sig2, amp3, mu3, sig3):
	return gaussian(x, amp1, mu1, sig1) + gaussian(x, amp2, mu2, sig2) + gaussian(x, amp3, mu3, sig3)

def fitEnergySpectrum(df, func, key, directory, filename, ext, bins, xlim, n_peaks, peaks_val, normalize=True):
# vim-marker--->
	data = df[key].dropna().to_numpy()

	counts, bin_edges = np.histogram(data, bins=bins, range=xlim, density=normalize)
	bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	max_ind = np.argmax(counts)
	max_pos = bin_centers[max_ind]

	data *= peaks_val[0] / max_pos

	new_xlim = (1.8, 10)
	counts, bin_edges = np.histogram(data, bins=bins, range=new_xlim, density=normalize)
	bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

	p0 = [
		0.80 * max(counts), peaks_val[0], 0.7,
		0.13 * max(counts), peaks_val[1], 0.2,
		0.13 * max(counts), peaks_val[2], 0.2
	]

	try:
		popt, pcov = curve_fit(
			func, bin_centers, counts, p0=p0,
			bounds=(
				[
					0.25, 5.8, 0.01,
					0, 6, 0.01,
					0, 2.5, 0.0
				], [
					np.inf, 6, np.inf,
					0.1, 7, np.inf,
					np.inf, 3.5, np.inf
				]
			),
			maxfev=10000
		)
	except RuntimeError as e:
		print(f"Fit error: {e}")
		popt = [0] * len(p0)

	plt.figure(figsize=(8, 5))
	plt.hist(data, bins=bins, range=new_xlim, histtype="step", density=normalize)

	x_fit = np.linspace(new_xlim[0], new_xlim[1], 1000)
	y_fit = func(x_fit, *popt)
	plt.plot(x_fit, y_fit, color="red")

	g1 = gaussian(x_fit, popt[0], popt[1], popt[2])
	g2 = gaussian(x_fit, popt[3], popt[4], popt[5])
	g3 = gaussian(x_fit, popt[6], popt[7], popt[8])

	plt.fill_between(x_fit, g1, color="blue",   alpha=0.3)
	plt.fill_between(x_fit, g2, color="green",  alpha=0.3)
	plt.fill_between(x_fit, g3, color="orange", alpha=0.3)

	plt.xlabel("Energy [keV]")
	plt.ylabel("Counts (normalized)" if normalize else "Counts")

	plt.tight_layout()
	plt.savefig(f"{directory}/{filename}.{ext}")
	plt.close()
# <---vim-marker


def calibLinFit(frame, func, directory, filename, ext, draw=False):
# vim-marker--->
	frame = frame[(frame["PileUp"] == 0) & (frame["OverFlow"] == 0)]

	asic_ids = sorted(frame["ASIC"].unique())
	channel_ids = sorted(frame["Channel"].unique())

	num_asics = len(asic_ids)
	num_channels = len(channel_ids)
	total = num_asics * num_channels
	step = 0

	A = np.full((num_asics, num_channels), np.nan)
	B = np.full((num_asics, num_channels), np.nan)

	print("\nStarting calibration")
	for i, asic in enumerate(asic_ids):
		for j, ch in enumerate(channel_ids):
			step += 1
			percent = (step / total) * 100
			print(f"Progress: {percent:.2f}% ({step}/{total})", end="\r")

			subset = frame[(frame["ASIC"] == asic) & (frame["Channel"] == ch)]

			if subset.empty:
				continue

			xdata = subset["iCal"].values
			ydata = subset["Adc"].values

			try:
				popt, _ = curve_fit(func, xdata, ydata)
				a, b = popt

				A[i, j] = a
				B[i, j] = b

				if draw:
					plt.figure()
					plt.scatter(xdata, ydata, s=1, alpha=0.2)
					plt.plot(np.sort(xdata), func(np.sort(xdata), *popt), color="red")
					plt.xlabel("iCal")
					plt.ylabel("Adc")
					plt.grid(True)
					plt.tight_layout()
					plt.savefig(f"{directory}/{filename}_ASIC_{asic}_CH_{ch}.{ext}")
					plt.close()

			except RuntimeError:
				print(f"\nFit error: ASIC {asic}, Channel {ch}")

	print("\nDone.")

# <---vim-marker
	return A, B

@njit
def compute_iCal_numba(A, B, adc, asic_idx, channel_idx, valid_mask):
# vim-marker--->
	n = len(adc)
	a = np.full(n, np.nan)
	b = np.full(n, np.nan)
	iCal = np.full(n, np.nan)

	for i in range(n):
		if valid_mask[i]:
			ai = asic_idx[i]
			cj = channel_idx[i]

			a_val = A[ai, cj]
			b_val = B[ai, cj]

			if not np.isnan(a_val) and a_val != 0:
				a[i] = a_val
				b[i] = b_val
				iCal[i] = (adc[i] - b_val) / a_val

# <---vim-marker
	return iCal

def iCalRecon(df, A, B, xKey, yKey, vKey, show_progress=True):
# vim-marker--->
	asic_ids = sorted(df[xKey].unique())
	channel_ids = sorted(df[yKey].unique())

	asic_to_idx = {asic: idx for idx, asic in enumerate(asic_ids)}
	channel_to_idx = {ch: idx for idx, ch in enumerate(channel_ids)}

	asic_idx = df[xKey].map(asic_to_idx).to_numpy()
	channel_idx = df[yKey].map(channel_to_idx).to_numpy()

	a = np.full(len(df), np.nan)
	b = np.full(len(df), np.nan)

	valid_mask = (~pd.isna(asic_idx)) & (~pd.isna(channel_idx))
	valid_mask &= (asic_idx >= 0) & (channel_idx >= 0)

	asic_idx_valid = asic_idx[valid_mask].astype(int)
	channel_idx_valid = channel_idx[valid_mask].astype(int)

	chunk_size = 100_000
	total = np.sum(valid_mask)
	processed = 0

	print()
	for start in range(0, total, chunk_size):
		end = min(start + chunk_size, total)
		idx_range = np.arange(len(valid_mask))[valid_mask][start:end]
		a[idx_range] = A[asic_idx_valid[start:end], channel_idx_valid[start:end]]
		b[idx_range] = B[asic_idx_valid[start:end], channel_idx_valid[start:end]]

		if show_progress:
			processed += (end - start)
			percent = (processed / total) * 100
			print(f"Mapping A, B... {percent:.2f}% ({processed}/{total})", end="\r")

	compute_mask = valid_mask & (~np.isnan(a)) & (a != 0)

	iCal = np.full(len(df), np.nan)
	iCal[compute_mask] = (df.loc[compute_mask, "Adc"].to_numpy() - b[compute_mask]) / a[compute_mask]

	df = df.copy()
	df[vKey] = iCal

	if show_progress:
		print("\nReconstruction complete.")

# <---vim-marker
	return df


def plotMap(df, xKey, yKey, vKey, directory, filename, ext, cmap="viridis", vmin=None, vmax=None, plot_features=False, normalize = False):
# vim-marker--->
	xdata_ids = sorted(df[xKey].unique())
	ydata_ids = sorted(df[yKey].unique())

	num_xdata = len(xdata_ids)
	num_ydata = len(ydata_ids)

	xdata_idx_map = {asic: i for i, asic in enumerate(xdata_ids)}
	ydata_idx_map = {ch: i for i, ch in enumerate(ydata_ids)}

	matrix = np.full((num_ydata, num_xdata), np.nan)

	grouped = df.groupby([xKey, yKey])[vKey].mean()

	if normalize:
		max_val = grouped.max()
		if max_val != 0:
			grouped = grouped / max_val

	for (asic, channel), value in grouped.items():
		i = ydata_idx_map[channel]
		j = xdata_idx_map[asic]
		matrix[i, j] = value

	plt.figure(figsize=(10, 6))
	im = plt.imshow(matrix, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

	plt.colorbar(im, label=vKey)
	if plot_features:
		plt.xticks(ticks=np.arange(num_xdata), labels=xdata_ids, rotation=90)
		plt.yticks(ticks=np.arange(num_ydata), labels=ydata_ids)
		plt.grid(True)
	else:
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)

	plt.xlabel(xKey)
	plt.ylabel(yKey)
	plt.tight_layout()
	plt.savefig(f"{directory}/{filename}.{ext}")
	plt.close()
# <---vim-marker


@njit
def compute_event_groups(time_data, prev_last_time, prev_max_group):
# vim-marker--->
	n = len(time_data)
	group_id = np.empty(n, dtype=np.int64)

	group = prev_max_group + 1
	for i in range(n):
		if i == 0:
			if prev_last_time is not None and time_data[i] - prev_last_time >= 100:
				group += 1
		else:
			if time_data[i] - time_data[i - 1] >= 100:
				group += 1
		group_id[i] = group

# <---vim-marker
	return group_id, group, time_data[-1]


@njit
def compute_coord_and_axis(asic_arr, channel_arr):
# vim-marker--->
	n = len(asic_arr)
	coord = np.empty(n, dtype=np.int64)
	axis = np.zeros(n, dtype=np.int64)

	for i in range(n):
		adj = asic_arr[i] - 86
		ch = channel_arr[i]
		if adj > 3:
			coord[i] = ch + (3 - adj) * 32
			axis[i] = 1
		else:
			coord[i] = 31 - ch + adj * 32
			axis[i] = 0
# <---vim-marker
	return coord, axis


@njit
def compute_event_stats(event_ids, coords, axes, ical):
# vim-marker--->
	max_event_id = np.max(event_ids) + 1

	x_sum_w = np.zeros(max_event_id)
	x_sum_wc = np.zeros(max_event_id)

	y_sum_w = np.zeros(max_event_id)
	y_sum_wc = np.zeros(max_event_id)

	sum_val = np.zeros(max_event_id)

	for i in range(len(event_ids)):
		eid = event_ids[i]
		w = ical[i]
		c = coords[i]
		ax = axes[i]

		sum_val[eid] += w

		if ax == 0:
			x_sum_w[eid] += w
			x_sum_wc[eid] += w * c
		else:
			y_sum_w[eid] += w
			y_sum_wc[eid] += w * c

	x_mean = np.full(max_event_id, np.nan)
	y_mean = np.full(max_event_id, np.nan)

	for eid in range(max_event_id):
		if x_sum_w[eid] > 0:
			x_mean[eid] = np.round(x_sum_wc[eid] / x_sum_w[eid])
		if y_sum_w[eid] > 0:
			y_mean[eid] = np.round(y_sum_wc[eid] / y_sum_w[eid])

# <---vim-marker
	return x_mean, y_mean, sum_val


def mapEvents(df, xKey, yKey, vKey, tKey):
# vim-marker--->
	df = df.sort_values("TimeStamp", ignore_index=True)

	chunk_size = 100_000
	total_len = len(df)

	prev_max_group = -1
	prev_last_time = None

	event_id_list = []
	coord_list = []
	axis_list = []
	ical_list = []

	print("\nMapping events...")

	for start_idx in tqdm(range(0, total_len, chunk_size), desc="Processing chunks"):
		end_idx = min(start_idx + chunk_size, total_len)
		chunk = df.iloc[start_idx:end_idx]

		time_data = chunk[tKey].to_numpy(dtype=np.int64)
		asic_arr = chunk[xKey].to_numpy(dtype=np.int64)
		channel_arr = chunk[yKey].to_numpy(dtype=np.int64)
		ical = chunk[vKey].to_numpy(dtype=np.float64)

		group_id, prev_max_group, prev_last_time = compute_event_groups(
			time_data, prev_last_time, prev_max_group
		)
		coord, axis = compute_coord_and_axis(asic_arr, channel_arr)

		event_id_list.append(group_id)
		coord_list.append(coord)
		axis_list.append(axis)
		ical_list.append(ical)

	print("\nComputing event-level statistics...")

	event_ids = np.concatenate(event_id_list)
	coords = np.concatenate(coord_list)
	axes = np.concatenate(axis_list)
	ical = np.concatenate(ical_list)

	event_x, event_y, event_val = compute_event_stats(event_ids, coords, axes, ical)

	event_df = pd.DataFrame({
		"EventID": np.arange(len(event_x)),
		"axX": event_x,
		"axY": event_y,
		"Val": event_val
	})

	print("Event mapping complete.\n")
# <---vim-marker
	return event_df


def loadData(filename: str, sep: str = ",", ext: str = "gz") -> pd.DataFrame:
# vim-marker--->
	is_gz = filename.endswith(ext)

	if is_gz:
		with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as f:
			total_lines = sum(1 for _ in f)
	else:
		with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
			total_lines = sum(1 for _ in f)

	chunk_size = 100_000
	reader = pd.read_csv(
		filename,
		sep=sep,
		compression='gzip' if is_gz else None,
		chunksize=chunk_size,
		iterator=True,
		low_memory=False
	)

# <---vim-marker
	return pd.concat(tqdm(reader, total=total_lines // chunk_size + 1, desc=f"Loading {filename}"))


def loadNpz(filename: str) -> dict:
# vim-marker--->
	data = np.load(filename)
	keys = list(data.keys())

	result = {}
	for key in tqdm(keys, desc=f"Loading {filename}"):
		result[key] = data[key]

	data.close()
# <---vim-marker
	return result


if __name__ == "__main__":
	raw_df = loadData(
		"../data/raw_data_ical_no_calib.dat.gz",
		sep="\t", ext="gz"
	)

	plain_df = loadData(
		"../data/GEM_HV_3930_2_AGH_plain_part_009.pcap_raw_hits.dat.gz",
		sep="\t", ext="gz"
	)

	AGH_df = loadData(
		"../data/GEM_HV_3930_2_AGH_UST_2_part_009.pcap_raw_hits.dat.gz",
		sep="\t", ext="gz"
	)

	plain_np = loadNpz("../data/GEM_HV_3930_2_AGH_plain.npz")

	A, B = calibLinFit(
		raw_df, linearFunc,
		"plots", "calib", "pdf",
		True
	)

	mapped_plain_df = mapEvents(plain_df, "ASIC", "Channel", "Adc", "TimeStamp")
	plotMap(
		mapped_plain_df, "axX", "axY", "Val",
		"plots", "GG_map_plain", "pdf",
		"RdYlGn_r", 0.5, 1.0,
		plot_features = False,
		normalize = True
	)

	fitEnergySpectrum(
		mapped_plain_df, tripleGaussian, "Val",
		"plots", "spectrum_mapped_plain_rec", "pdf",
		100, (0, 3000), 3, np.array([5.9, 6.49, 3]),
		True
	)

	mapped_AGH_df = mapEvents(AGH_df, "ASIC", "Channel", "Adc", "TimeStamp")
	plotMap(
		mapped_plain_df, "axX", "axY", "Val",
		"plots", "GG_map_AGH", "pdf",
		"RdYlGn_r", 0.5, 1.0,
		plot_features = False,
		normalize = True
	)


	plain_rec = iCalRecon(
		plain_df, A, B,
		"ASIC", "Channel", "iCalRecon"
	)

	AGH_rec = iCalRecon(
		AGH_df, A, B,
		"ASIC", "Channel", "iCalRecon"
	)


	plotMap(
		plain_rec, "ASIC", "Channel", "iCalRecon",
		"plots", "recPlainMap", "pdf",
		"viridis", 50, 350,
		plot_features = True,
		normalize = False
	)


	mapped_plain_rec = mapEvents(plain_rec, "ASIC", "Channel", "iCalRecon", "TimeStamp")
	plotMap(
		mapped_plain_rec, "axX", "axY", "Val",
		"plots", "mappedPlainRec", "pdf",
		"RdYlGn_r",
		plot_features = False,
		normalize = True
	)

	mapped_AGH_rec = mapEvents(AGH_rec, "ASIC", "Channel", "iCalRecon", "TimeStamp")
	plotMap(
		mapped_AGH_rec, "axX", "axY", "Val",
		"plots", "mappedAGHRec", "pdf",
		"RdYlGn_r",
		plot_features = False,
		normalize = True
	)

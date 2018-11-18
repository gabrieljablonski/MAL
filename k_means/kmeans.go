package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const(
	tolerance = 0.001
	maxClusters = 20
)

var nClusters = 1

type Driver struct {
	distance float64
	speed float64
}

type Point struct {
	x1 float64
	x2 float64
}

type Cluster struct {
	centroid Point
	drivers []Driver
}

func (c Cluster) moveCentroid() Point{
	newX1, newX2 := 0.0, 0.0
	for _, driver := range c.drivers {
		newX1 += driver.distance
		newX2 += driver.speed
	}
	newX1, newX2 = newX1/float64(len(c.drivers)), newX2/float64(len(c.drivers))

	return Point {newX1, newX2}
}

func (c Cluster) getAverageIntraClusterDistance() float64{
	if c.drivers == nil{
		return 0.0
	}
	totalDistance := 0.0
	for _, driver := range c.drivers {
		totalDistance += math.Sqrt(math.Pow(driver.distance-c.centroid.x1, 2) +
			math.Pow(driver.speed-c.centroid.x2, 2))
	}
	return totalDistance/float64(len(c.drivers))
}

func getAverageDistance(clusters []Cluster) float64 {
	totalDistance := 0.0
	for _, cluster := range clusters {
		totalDistance += cluster.getAverageIntraClusterDistance()
	}
	return 	totalDistance/float64(nClusters)
}

func mapTo(x float64, lim [2]float64) float64 {
	min := math.Min(lim[0], lim[1])
	max := math.Max(lim[0], lim[1])
	return x*(max - min) + min
}

func loadDrivers(filePath string) []Driver{
	bytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Print(err)
	}
	s := string(bytes)
	lines := strings.Split(s, "\n")[1:] // ignore header
	var drivers []Driver
	for _, line := range lines {
		split := strings.Split(line, "\t")

		if len(split) != 3{
			continue
		}

		distance, err := strconv.ParseFloat(split[1], 64)
		if err != nil{
			fmt.Print(err)
		}

		speed, err := strconv.ParseFloat(split[2], 64)
		if err != nil{
			fmt.Print(err)
		}

		drivers = append(drivers, Driver{distance, speed})
	}
	return drivers
}

func getLimits(drivers []Driver) ([2]float64, [2]float64){
	limX1 := [2]float64{math.MaxInt64, math.MinInt64}
	limX2 := [2]float64{math.MaxInt64, math.MinInt64}
	for _, driver := range drivers {
		if driver.distance < limX1[0]{
			limX1[0] = driver.distance
		}
		if driver.distance > limX1[1]{
			limX1[1] = driver.distance
		}
		if driver.speed < limX2[0]{
			limX2[0] = driver.speed
		}
		if driver.speed > limX2[1]{
			limX2[1] = driver.speed
		}
	}
	return limX1, limX2
}

func reorganizeDrivers(clusters []Cluster, drivers []Driver) []Cluster {
	for c := range clusters {
		clusters[c].drivers = nil
	}

	for _, driver := range drivers {
		closest, minDist := -1, float64(math.MaxInt64)
		for c, cluster := range clusters {
			dist := math.Sqrt(math.Pow(driver.distance-cluster.centroid.x1, 2) +
							  math.Pow(driver.speed-cluster.centroid.x2, 2))
			if dist <= minDist {
				closest, minDist = c, dist
			}
		}
		clusters[closest].drivers = append(clusters[closest].drivers, driver)
	}
	return clusters
}

func initializeClusters(nClusters int, drivers []Driver) []Cluster {
	clusters := make([]Cluster, nClusters)

	limX1, limX2 := getLimits(drivers)
	centroids := initializeCentroids(nClusters, limX1, limX2)

	for i, centroid := range centroids {
		clusters[i].centroid = centroid
	}

	clusters = reorganizeDrivers(clusters, drivers)

	return clusters
}

func initializeCentroids(nClusters int, limX1 [2]float64, limX2 [2]float64) []Point {
	centroids := make([]Point, nClusters)

	for i := 0; i < nClusters; i++ {
		x1 := mapTo(rand.Float64(), limX1)
		x2 := mapTo(rand.Float64(), limX2)

		centroids[i] = Point{x1, x2}
	}
	return centroids
}

func writeToFile(filePath string, clusters []Cluster) {
	file, err := os.Create(filePath)
	check(err)
	defer file.Close()

	w := bufio.NewWriter(file)

	for c, cluster := range clusters {

		x1, x2 := cluster.centroid.x1, cluster.centroid.x2
		if math.IsNaN(x1) {
			x1 = 0.0
		}
		if math.IsNaN(x2) {
			x2 = 0.0
		}
		_, err = fmt.Fprintf(w, fmt.Sprintf("%d\t%.5f\t%.5f\n", c, x1, x2))
		check(err)
	}

	for c, cluster := range clusters {
		for _, driver := range cluster.drivers {
			distance, speed := driver.distance, driver.speed
			_, err = fmt.Fprintf(w, fmt.Sprintf("%d\t%.5f\t%.5f\n", c, distance, speed))
			check(err)
		}
	}
	w.Flush()
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func init(){
	seed := time.Now().UnixNano()
	rand.Seed(seed)
}

func main(){
	filePath := "data_1024.csv"
	drivers := loadDrivers(filePath)

	file, err := os.Create("average_distance_by_k.txt")
	check(err)
	defer file.Close()

	w := bufio.NewWriter(file)

	for i := 1; i <= maxClusters; i++ {
		nClusters = i
		clusters := initializeClusters(nClusters, drivers)

		fmt.Printf("Clusters: %d\n-------------------\n", nClusters)
		lastDistance := 0.0
		for cycle := 1; ; cycle++ {
			totalDistance := getAverageDistance(clusters)

			if math.IsNaN(totalDistance) {
				break
			}

			fmt.Printf("Cycle: %d\tAverage Distance: %.5f\n", cycle, totalDistance)

			delta := math.Abs(totalDistance - lastDistance)
			if delta < tolerance*totalDistance {
				break
			}
			lastDistance = totalDistance

			newClusters := make([]Cluster, nClusters)

			for c, cluster := range clusters {
				cluster.centroid = cluster.moveCentroid()
				newClusters[c] = cluster
			}
			copy(clusters, newClusters)
			reorganizeDrivers(clusters, drivers)

		}
		fmt.Printf("-------------------\n")

		_, err = fmt.Fprintf(w, fmt.Sprintf("%d\t%.5f\n", nClusters, getAverageDistance(clusters)))
		check(err)

		writeToFile(fmt.Sprintf("clustered/%d.txt", nClusters), clusters)
	}
	w.Flush()
}

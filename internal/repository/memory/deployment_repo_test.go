package memory_test

import (
	"errors"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newDeploymentRepo() (*memory.DeploymentRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewDeploymentRepo(s), s
}

func TestDeploymentRepo_Create(t *testing.T) {
	t.Parallel()

	r, s := newDeploymentRepo()
	dep := &domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}}

	err := r.Create(dep)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(s.Deployments["task_1"]) != 1 {
		t.Errorf("expected 1 deployment, got %d", len(s.Deployments["task_1"]))
	}
}

func TestDeploymentRepo_Get_Found(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	got, err := r.Get("dep_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if got.ID != "dep_1" {
		t.Errorf("expected ID 'dep_1', got %q", got.ID)
	}
}

func TestDeploymentRepo_Get_NotFound(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()

	_, err := r.Get("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentRepo_Get_AcrossTasks(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	_ = r.Create(&domain.Deployment{ID: "dep_2", TaskID: "task_2", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	got, err := r.Get("dep_2")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if got.TaskID != "task_2" {
		t.Errorf("expected TaskID 'task_2', got %q", got.TaskID)
	}
}

func TestDeploymentRepo_GetActiveForTask_Found(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentStopped, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	_ = r.Create(&domain.Deployment{ID: "dep_2", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	got, err := r.GetActiveForTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if got.ID != "dep_2" {
		t.Errorf("expected active deployment dep_2, got %q", got.ID)
	}
}

func TestDeploymentRepo_GetActiveForTask_LatestActive(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	_ = r.Create(&domain.Deployment{ID: "dep_2", TaskID: "task_1", Status: domain.DeploymentStopped, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	_ = r.Create(&domain.Deployment{ID: "dep_3", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	got, err := r.GetActiveForTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if got.ID != "dep_3" {
		t.Errorf("expected latest active deployment dep_3, got %q", got.ID)
	}
}

func TestDeploymentRepo_GetActiveForTask_NoneActive(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentStopped, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	_, err := r.GetActiveForTask("task_1")
	if !errors.Is(err, domain.ErrNoDeployment) {
		t.Errorf("expected ErrNoDeployment, got %v", err)
	}
}

func TestDeploymentRepo_GetActiveForTask_NoDeployments(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()

	_, err := r.GetActiveForTask("task_1")
	if !errors.Is(err, domain.ErrNoDeployment) {
		t.Errorf("expected ErrNoDeployment, got %v", err)
	}
}

func TestDeploymentRepo_ListByTask(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	_ = r.Create(&domain.Deployment{ID: "dep_2", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 deployments, got %d", len(list))
	}
}

func TestDeploymentRepo_ListByTask_Empty(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 0 {
		t.Errorf("expected 0 deployments, got %d", len(list))
	}
}

func TestDeploymentRepo_ListByTask_ReturnsCopy(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	list1, _ := r.ListByTask("task_1")
	list1[0] = nil

	list2, _ := r.ListByTask("task_1")
	if list2[0] == nil {
		t.Error("expected ListByTask to return a copy, not internal slice")
	}
}

func TestDeploymentRepo_Update_Found(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()
	_ = r.Create(&domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})

	updated := &domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentStopped, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}}

	err := r.Update(updated)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("dep_1")
	if got.Status != domain.DeploymentStopped {
		t.Errorf("expected status stopped, got %v", got.Status)
	}
}

func TestDeploymentRepo_Update_NotFound(t *testing.T) {
	t.Parallel()

	r, _ := newDeploymentRepo()

	err := r.Update(&domain.Deployment{ID: "missing", TaskID: "task_1", TrainingJobID: "", Endpoint: "", Autoscale: false, Status: "", RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}})
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
